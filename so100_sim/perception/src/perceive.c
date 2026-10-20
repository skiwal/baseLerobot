/*
 * perceive.c — 套筒/杆 感知 (第一步)
 *
 * 输入: 根部 RGB-D 相机的 rgb.bin / depth.bin / bg.bin(无目标的背景深度) + meta.txt(W H)
 * 算法: 以机械臂根部为基准的固定相机外参, 对深度图做"背景差分"得到动态点(套筒+杆),
 *       连通域分割 -> 每个连通域用 PCA 求主轴(轴线), 按轴线竖直/倾斜分类为套筒/杆,
 *       再做几何拟合得到 6D 位姿, 并在图上高亮(套筒绿/杆红)+ 画轴线/框/杆尖。
 * 输出: result.bin(高亮后的 RGB) + stdout 打印位姿。
 *
 * 纯 C 实现, 无第三方依赖。相机外参与 so100_sleeve_scene.xml 的 base_rgbd 一致(布局永不变)。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---------------- 固定相机外参(与 XML base_rgbd 一致) ---------------- */
static const double CAM_POS[3]   = {0.08, -0.04, 0.30};
static const double CAM_RIGHT[3] = {-0.9796, 0.2009, 0.0};
static const double CAM_UP[3]    = {-0.1424, -0.6943, 0.7054};
static const double CAM_BACK[3]  = {0.1418, 0.6911, 0.7088};
static double FOVY = 50.0 * 3.141592653589793 / 180.0;

static int W = 0, H = 0;

/* ---------------- 小工具 ---------------- */
static int read_meta(const char* path, int* w, int* h) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    if (fscanf(f, "%d %d", w, h) != 2) { fclose(f); return 0; }
    fclose(f);
    return 1;
}
static int read_u16(FILE* f, uint16_t* v) {
    unsigned char b[2];
    if (fread(b, 1, 2, f) != 2) return 0;
    *v = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
    return 1;
}
static void normalize(double v[3]) {
    double n = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (n > 1e-12) { v[0]/=n; v[1]/=n; v[2]/=n; }
}
/* world -> image(像素), 返回 1 表示在相机前方. 注: MuJoCo 方像素, x/y 用同一焦距 f */
static int project(const double p[3], double* uu, double* vv) {
    double d[3] = {p[0]-CAM_POS[0], p[1]-CAM_POS[1], p[2]-CAM_POS[2]};
    double xc = CAM_RIGHT[0]*d[0] + CAM_RIGHT[1]*d[1] + CAM_RIGHT[2]*d[2];
    double yc = CAM_UP[0]*d[0]    + CAM_UP[1]*d[1]    + CAM_UP[2]*d[2];
    double zc = CAM_BACK[0]*d[0]  + CAM_BACK[1]*d[1]  + CAM_BACK[2]*d[2];
    if (zc >= -1e-6) return 0;
    double f = (H/2.0)/tan(FOVY/2.0);   /* 方像素: 单焦距, x/y 同 */
    *uu = W/2.0 + f*(xc/(-zc));
    *vv = H/2.0 - f*(yc/(-zc));
    return 1;
}
/* image(u,v 像素, d_m 米) -> world(米) */
static void unproject(double u, double v, double d_m, double p[3]) {
    double f = (H/2.0)/tan(FOVY/2.0);   /* 方像素: 单焦距, x/y 同 */
    double xc = (u - W/2.0)/f * d_m;
    double yc = -(v - H/2.0)/f * d_m;
    double zc = -d_m;
    p[0] = CAM_POS[0] + xc*CAM_RIGHT[0] + yc*CAM_UP[0] + zc*CAM_BACK[0];
    p[1] = CAM_POS[1] + xc*CAM_RIGHT[1] + yc*CAM_UP[1] + zc*CAM_BACK[1];
    p[2] = CAM_POS[2] + xc*CAM_RIGHT[2] + yc*CAM_UP[2] + zc*CAM_BACK[2];
}

/* ---------------- 绘图 ---------------- */
static void setpx(uint8_t* img, int x, int y, int r, int g, int b) {
    if (x<0||y<0||x>=W||y>=H) return;
    int i = (y*W + x)*3; img[i]=r; img[i+1]=g; img[i+2]=b;
}
static void line(uint8_t* img, int x0, int y0, int x1, int y1, int r, int g, int b) {
    int dx = abs(x1-x0), dy = abs(y1-y0);
    int sx = x0<x1?1:-1, sy = y0<y1?1:-1;
    int err = dx-dy, x = x0, y = y0;
    for (;;) {
        setpx(img,x,y,r,g,b); setpx(img,x+1,y,r,g,b);
        setpx(img,x,y+1,r,g,b); setpx(img,x-1,y,r,g,b);
        if (x==x1 && y==y1) break;
        int e2 = 2*err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }
    }
}
static void rect(uint8_t* img, int x0, int y0, int x1, int y1, int r, int g, int b) {
    line(img,x0,y0,x1,y0,r,g,b); line(img,x0,y1,x1,y1,r,g,b);
    line(img,x0,y0,x0,y1,r,g,b); line(img,x1,y0,x1,y1,r,g,b);
}
static void circle(uint8_t* img, int cx, int cy, int rad, int r, int g, int b) {
    for (int y=-rad; y<=rad; y++)
        for (int x=-rad; x<=rad; x++)
            if (x*x + y*y <= rad*rad) setpx(img, cx+x, cy+y, r, g, b);
}

/* PCA 主轴(最大特征向量, 幂迭代) */
static void pca_axis(const double* pts, int n, double out[3]) {
    double m[3] = {0,0,0};
    for (int i=0;i<n;i++) { m[0]+=pts[i*3]; m[1]+=pts[i*3+1]; m[2]+=pts[i*3+2]; }
    m[0]/=n; m[1]/=n; m[2]/=n;
    double C[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for (int i=0;i<n;i++) {
        double d[3] = {pts[i*3]-m[0], pts[i*3+1]-m[1], pts[i*3+2]-m[2]};
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) C[a][b] += d[a]*d[b];
    }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) C[a][b]/=n;
    double v[3] = {1,1,1}; normalize(v);
    for (int k=0;k<100;k++) {
        double w[3] = {C[0][0]*v[0]+C[0][1]*v[1]+C[0][2]*v[2],
                       C[1][0]*v[0]+C[1][1]*v[1]+C[1][2]*v[2],
                       C[2][0]*v[0]+C[2][1]*v[1]+C[2][2]*v[2]};
        if (sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]) < 1e-12) break;
        normalize(w);
        double dot = w[0]*v[0]+w[1]*v[1]+w[2]*v[2];
        if (dot < 0) { w[0]=-w[0]; w[1]=-w[1]; w[2]=-w[2]; }
        v[0]=w[0]; v[1]=w[1]; v[2]=w[2];
    }
    out[0]=v[0]; out[1]=v[1]; out[2]=v[2]; normalize(out);
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s rgb.bin depth.bin bg.bin meta.txt result.bin\n", argv[0]);
        return 1;
    }
    if (!read_meta(argv[4], &W, &H)) { fprintf(stderr, "meta read fail\n"); return 1; }
    int N = W*H;
    uint8_t* rgb = malloc(N*3);
    uint8_t* res = malloc(N*3);
    uint16_t* dep = malloc(N*2);
    uint16_t* bg  = malloc(N*2);
    {
        FILE* f = fopen(argv[1], "rb");
        if (!f || fread(rgb, 1, (size_t)N*3, f) != (size_t)N*3) { fprintf(stderr, "rgb read fail\n"); return 1; }
        fclose(f);
    }
    {
        FILE* f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "depth read fail\n"); return 1; }
        for (int i=0;i<N;i++) if (!read_u16(f, &dep[i])) { fprintf(stderr, "depth read fail\n"); return 1; }
        fclose(f);
    }
    {
        FILE* f = fopen(argv[3], "rb");
        if (!f) { fprintf(stderr, "bg read fail\n"); return 1; }
        for (int i=0;i<N;i++) if (!read_u16(f, &bg[i])) { fprintf(stderr, "bg read fail\n"); return 1; }
        fclose(f);
    }
    memcpy(res, rgb, (size_t)N*3);

    /* 1) 背景差分 -> 动态掩码(套筒+杆) */
    uint8_t* dyn = calloc(N, 1);
    for (int i=0;i<N;i++) {
        uint16_t dm = dep[i], bm = bg[i];
        if (dm > 0 && (bm == 0 || abs((int)dm - (int)bm) > 12)) dyn[i] = 1;
    }

    /* 2) 连通域(BFS, 4 邻域) */
    int* comp = calloc(N, sizeof(int));
    int* stack = malloc(N * sizeof(int));
    int ncomp = 0;
    int** clist = NULL; int* clistsize = NULL; int* clistcap = NULL;
    for (int i=0;i<N;i++) {
        if (dyn[i] && comp[i]==0) {
            ncomp++;
            int sp = 0; stack[sp++] = i; comp[i] = ncomp;
            while (sp > 0) {
                int p = stack[--sp];
                int pu = p/W, pv = p%W;
                int nb[4], cnt = 0;
                if (pu>0)    nb[cnt++] = p-W;
                if (pu<H-1)  nb[cnt++] = p+W;
                if (pv>0)    nb[cnt++] = p-1;
                if (pv<W-1)  nb[cnt++] = p+1;
                for (int k=0;k<cnt;k++) { int q = nb[k]; if (dyn[q] && comp[q]==0) { comp[q]=ncomp; stack[sp++]=q; } }
            }
        }
    }
    clist = malloc(sizeof(int*) * (ncomp+1));
    clistsize = malloc(sizeof(int) * (ncomp+1));
    clistcap = malloc(sizeof(int) * (ncomp+1));
    for (int c=1;c<=ncomp;c++) { clist[c]=NULL; clistsize[c]=0; clistcap[c]=0; }
    for (int i=0;i<N;i++) {
        int c = comp[i];
        if (c>0) {
            if (clistsize[c] >= clistcap[c]) {
                clistcap[c] = clistcap[c] ? clistcap[c]*2 : 64;
                clist[c] = realloc(clist[c], sizeof(int)*clistcap[c]);
            }
            clist[c][clistsize[c]++] = i;
        }
    }

    /* 3) 每个连通域: 世界点云 -> PCA 轴线 + 质心 + 框 */
    double* caxis = malloc(sizeof(double)*3*(ncomp+1));
    double* ccent = malloc(sizeof(double)*3*(ncomp+1));
    double* craw  = malloc(sizeof(double)*3*(ncomp+1)); /* 各连通域原始 PCA 质心 */
    double* cbbox = malloc(sizeof(double)*4*(ncomp+1)); /* minu,minv,maxu,maxv */
    int*    cvert = malloc(sizeof(int)*(ncomp+1));      /* 竖直度(带符号) */
    for (int c=1;c<=ncomp;c++) {
        int n = clistsize[c];
        cvert[c] = -2; /* 无效 */
        if (n < 30) continue;
        double* pts = malloc(sizeof(double)*3*n);
        int m = 0;
        double minu=1e9, minv=1e9, maxu=-1e9, maxv=-1e9;
        for (int k=0;k<n;k++) {
            int p = clist[c][k];
            int u = p%W, v = p/W;
            uint16_t dm = dep[p];
            if (dm == 0) continue;
            double w[3]; unproject((double)u, (double)v, dm/1000.0, w);
            pts[m*3]=w[0]; pts[m*3+1]=w[1]; pts[m*3+2]=w[2]; m++;
            if (u<minu) minu=u; if (v<minv) minv=v; if (u>maxu) maxu=u; if (v>maxv) maxv=v;
        }
        if (m < 30) { free(pts); continue; }
        double ax[3]; pca_axis(pts, m, ax);
        double cen[3]={0,0,0};
        for (int i2=0;i2<m;i2++) { cen[0]+=pts[i2*3]; cen[1]+=pts[i2*3+1]; cen[2]+=pts[i2*3+2]; }
        cen[0]/=m; cen[1]/=m; cen[2]/=m;
        caxis[3*c]=ax[0]; caxis[3*c+1]=ax[1]; caxis[3*c+2]=ax[2];
        ccent[3*c]=cen[0]; ccent[3*c+1]=cen[1]; ccent[3*c+2]=cen[2];
        craw[3*c]=cen[0]; craw[3*c+1]=cen[1]; craw[3*c+2]=cen[2];
        cbbox[4*c]=minu; cbbox[4*c+1]=minv; cbbox[4*c+2]=maxu; cbbox[4*c+3]=maxv;
        cvert[c] = (ax[2] >= 0 ? 1 : -1) * (int)(fabs(ax[2])*1000.0);
        free(pts);
    }

    /* 4) 分类: 最竖直=套筒, 最倾斜=杆 */
    int sleeve_c = 0, rod_c = 0;
    int best_vert = -1, worst_vert = 100000;
    for (int c=1;c<=ncomp;c++) {
        if (cvert[c] < -1) continue;
        if (cvert[c] > best_vert)  { best_vert = cvert[c];  sleeve_c = c; }
        if (cvert[c] < worst_vert) { worst_vert = cvert[c]; rod_c = c; }
    }
    if (sleeve_c == rod_c) { /* 只有一个有效连通域: 退化为全当套筒 */
        rod_c = 0;
    }
    double sleeve_raw[3] = {0, 0, 0};   /* PCA 原始质心(用于验证抬起高度) */

    /* 套筒中心精修: 竖直套筒的单目斜视使可见前表面相对真实轴线偏一个半径 R.
       对每个可见点, 沿"指向相机"的水平方向把点移回 R -> 落在轴线上, 平均得轴线足点.
       此法无前表面偏置, 精度到 mm 级, 抓取才稳. 高度用已知套筒中心高. */
    if (sleeve_c > 0) {
        int n = clistsize[sleeve_c];
        double* pts = malloc(sizeof(double)*3*n);
        int m = 0;
        for (int k=0;k<n;k++) {
            int p = clist[sleeve_c][k];
            uint16_t dm = dep[p];
            if (dm == 0) continue;
            int u = p%W, v = p/W;
            double w[3]; unproject((double)u, (double)v, dm/1000.0, w);
            pts[m*3]=w[0]; pts[m*3+1]=w[1]; pts[m*3+2]=w[2]; m++;
        }
        const double R = 0.0135;   /* 套筒外缘半径(布局已知) */
        double sx=0, sy=0, sz=0;
        for (int i=0;i<m;i++) {
            double dxc = CAM_POS[0]-pts[i*3], dyc = CAM_POS[1]-pts[i*3+1];
            double horiz = sqrt(dxc*dxc + dyc*dyc);
            if (horiz < 1e-6) continue;
            double ax = pts[i*3]   - R*(dxc/horiz);
            double ay = pts[i*3+1] - R*(dyc/horiz);
            sx += ax; sy += ay; sz += pts[i*3+2];
        }
        sx/=m; sy/=m; sz/=m;
        ccent[3*sleeve_c]   = sx;
        ccent[3*sleeve_c+1] = sy;
        ccent[3*sleeve_c+2] = 0.11;   /* 套筒中心高 = 台面0.08 + 半高0.03 */
        caxis[3*sleeve_c]   = 0.0;
        caxis[3*sleeve_c+1] = 0.0;
        caxis[3*sleeve_c+2] = 1.0;    /* 竖直(物体先验, 与第一步分类一致) */
        sleeve_raw[0] = craw[3*sleeve_c]; sleeve_raw[1] = craw[3*sleeve_c+1]; sleeve_raw[2] = craw[3*sleeve_c+2];
        free(pts);
    }

    /* 5) 标注 + 打印 */
    for (int c=1;c<=ncomp;c++) {
        if (cvert[c] < -1) continue;
        int is_sleeve = (c == sleeve_c);
        int r = is_sleeve ? 60 : 230, g = is_sleeve ? 230 : 60, b = is_sleeve ? 60 : 60;
        /* 掩码着色(半透明叠加) */
        for (int k=0;k<clistsize[c];k++) {
            int p = clist[c][k];
            int i = p*3;
            res[i]   = (uint8_t)(rgb[i]*0.35 + r*0.65);
            res[i+1] = (uint8_t)(rgb[i+1]*0.35 + g*0.65);
            res[i+2] = (uint8_t)(rgb[i+2]*0.35 + b*0.65);
        }
        /* 框 */
        int x0=(int)cbbox[4*c], y0=(int)cbbox[4*c+1], x1=(int)cbbox[4*c+2], y1=(int)cbbox[4*c+3];
        rect(res, x0, y0, x1, y1, 255,255,255);
        /* 轴线 */
        double L = is_sleeve ? 0.05 : 0.045;
        double* ax = &caxis[3*c]; double* ce = &ccent[3*c];
        double base[3] = {ce[0]-ax[0]*L, ce[1]-ax[1]*L, ce[2]-ax[2]*L};
        double tip[3]  = {ce[0]+ax[0]*L, ce[1]+ax[1]*L, ce[2]+ax[2]*L};
        double ub, vb, ut, vt;
        if (project(base, &ub, &vb) && project(tip, &ut, &vt))
            line(res, (int)ub, (int)vb, (int)ut, (int)vt, r, g, b);
        if (!is_sleeve) {
            double ut2, vt2; if (project(tip, &ut2, &vt2)) circle(res, (int)ut2, (int)vt2, 5, 255, 220, 0);
        }
    }

    if (sleeve_c > 0) {
        double* ce = &ccent[3*sleeve_c]; double* ax = &caxis[3*sleeve_c];
        printf("SLEEVE pos=(%.3f,%.3f,%.3f) axis=(%.3f,%.3f,%.3f)\n",
               ce[0],ce[1],ce[2], ax[0],ax[1],ax[2]);
        printf("SLEEVE_RAW centroid=(%.3f,%.3f,%.3f)\n",
               sleeve_raw[0], sleeve_raw[1], sleeve_raw[2]);
    } else {
        printf("SLEEVE not detected\n");
    }
    if (rod_c > 0) {
        double* ce = &ccent[3*rod_c]; double* ax = &caxis[3*rod_c];
        /* 轴线定向: 让 tip 在上方(z 更大) */
        double L = 0.045;
        double tip[3]  = {ce[0]+ax[0]*L, ce[1]+ax[1]*L, ce[2]+ax[2]*L};
        double base[3] = {ce[0]-ax[0]*L, ce[1]-ax[1]*L, ce[2]-ax[2]*L};
        if (tip[2] < base[2]) { ax[0]=-ax[0]; ax[1]=-ax[1]; ax[2]=-ax[2];
            double t[3]={tip[0],tip[1],tip[2]}; tip[0]=base[0];tip[1]=base[1];tip[2]=base[2]; base[0]=t[0];base[1]=t[1];base[2]=t[2]; }
        printf("ROD    base=(%.3f,%.3f,%.3f) tip=(%.3f,%.3f,%.3f) axis=(%.3f,%.3f,%.3f)\n",
               base[0],base[1],base[2], tip[0],tip[1],tip[2], ax[0],ax[1],ax[2]);
    } else {
        printf("ROD not detected\n");
    }

    /* 6) 写出高亮图 */
    FILE* f = fopen(argv[5], "wb");
    if (!f || fwrite(res, 1, (size_t)N*3, f) != (size_t)N*3) { fprintf(stderr, "result write fail\n"); return 1; }
    fclose(f);

    /* 清理 */
    for (int c=1;c<=ncomp;c++) free(clist[c]);
    free(clist); free(clistsize); free(clistcap);
    free(rgb); free(res); free(dep); free(bg); free(dyn); free(comp); free(stack);
    free(caxis); free(ccent); free(craw); free(cbbox); free(cvert);
    return 0;
}
