#!/usr/bin/env python3
"""
Voxel Engine Testing Application
PySide6 + OpenGL voxel algorithm testing & visualization tool

Controls:  LMB = Orbit | RMB = Pan | Wheel = Zoom
Install:   pip install PySide6 PyOpenGL numpy
"""

import sys, math, time, random
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea, QMessageBox, QSplitter,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPalette, QColor
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *


# ================================================================
#  Perlin Noise
# ================================================================
class PerlinNoise:
    def __init__(self, seed=0):
        self.perm = list(range(256))
        random.Random(seed).shuffle(self.perm)
        self.perm *= 2

    @staticmethod
    def _fade(t): return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(t, a, b): return a + t * (b - a)

    @staticmethod
    def _grad(h, x, y, z):
        h &= 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h in (12, 14) else z)
        return (-u if h & 1 else u) + (-v if h & 2 else v)

    def noise3d(self, x, y, z):
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        Z = int(math.floor(z)) & 255
        x -= math.floor(x); y -= math.floor(y); z -= math.floor(z)
        u, v, w = self._fade(x), self._fade(y), self._fade(z)
        p = self.perm
        A = p[X]+Y;   AA = p[A]+Z;   AB = p[A+1]+Z
        B = p[X+1]+Y; BA = p[B]+Z;   BB = p[B+1]+Z
        return self._lerp(w,
            self._lerp(v,
                self._lerp(u, self._grad(p[AA],x,y,z),     self._grad(p[BA],x-1,y,z)),
                self._lerp(u, self._grad(p[AB],x,y-1,z),   self._grad(p[BB],x-1,y-1,z))),
            self._lerp(v,
                self._lerp(u, self._grad(p[AA+1],x,y,z-1),   self._grad(p[BA+1],x-1,y,z-1)),
                self._lerp(u, self._grad(p[AB+1],x,y-1,z-1), self._grad(p[BB+1],x-1,y-1,z-1))))

    def octave(self, x, y, z, octaves=4, persistence=0.5, lacunarity=2.0):
        total = max_a = 0.0; freq = 1.0; amp = 1.0
        for _ in range(octaves):
            total  += self.noise3d(x*freq, y*freq, z*freq) * amp
            max_a  += amp
            amp    *= persistence
            freq   *= lacunarity
        return total / max_a


# ================================================================
#  Voxel Chunk  – dense array + face-culled mesh builder
# ================================================================
class VoxelChunk:
    AIR=0; STONE=1; DIRT=2; GRASS=3; SAND=4; WATER=5; WOOD=6; LEAVES=7; SNOW=8; CUSTOM=255

    BLOCK_COLORS = {
        0:(0,0,0,0), 1:(.55,.55,.55,1), 2:(.60,.40,.20,1), 3:(.25,.75,.22,1),
        4:(.90,.88,.55,1), 5:(.22,.42,.90,.7), 6:(.50,.30,.10,1),
        7:(.15,.58,.12,1), 8:(.93,.93,1.,1), 255:(1.,0.,1.,1),
    }

    _FACES = [
        ((1,0,0),  [(1,0,0),(1,1,0),(1,1,1),(1,0,1)]),
        ((-1,0,0), [(0,0,1),(0,1,1),(0,1,0),(0,0,0)]),
        ((0,1,0),  [(0,1,1),(1,1,1),(1,1,0),(0,1,0)]),
        ((0,-1,0), [(0,0,0),(1,0,0),(1,0,1),(0,0,1)]),
        ((0,0,1),  [(1,0,1),(0,0,1),(0,1,1),(1,1,1)]),
        ((0,0,-1), [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]),
    ]
    _NBR = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    def __init__(self, size=32):
        self.size = size
        self.data = np.zeros((size, size, size), dtype=np.uint8)
        self.colors: dict[tuple, tuple] = {}
        self.mesh_dirty = True
        self.vertices   = np.zeros(0, dtype=np.float32)
        self.normals    = np.zeros(0, dtype=np.float32)
        self.colors_arr = np.zeros(0, dtype=np.float32)
        self.vertex_count = 0
        self.face_count   = 0

    def get(self, x, y, z):
        if 0<=x<self.size and 0<=y<self.size and 0<=z<self.size:
            return self.data[x,y,z]
        return 0

    def set(self, x, y, z, v):
        if 0<=x<self.size and 0<=y<self.size and 0<=z<self.size:
            self.data[x,y,z] = v; self.mesh_dirty = True

    def get_color(self, x, y, z):
        if (x,y,z) in self.colors: return self.colors[(x,y,z)]
        return self.BLOCK_COLORS.get(int(self.data[x,y,z]), (1,0,1,1))

    def set_color(self, x, y, z, c):
        if self.data[x,y,z]: self.colors[(x,y,z)] = c; self.mesh_dirty = True

    def clear(self):
        self.data.fill(0); self.colors.clear(); self.mesh_dirty = True

    def count_voxels(self): return int(np.count_nonzero(self.data))

    def build_mesh(self):
        V, N, C = [], [], []
        s = self.size
        for idx in np.argwhere(self.data != 0):
            x, y, z = int(idx[0]), int(idx[1]), int(idx[2])
            col = self.get_color(x, y, z)
            for fi, (nrm, fv) in enumerate(self._FACES):
                nx2 = x+self._NBR[fi][0]; ny2 = y+self._NBR[fi][1]; nz2 = z+self._NBR[fi][2]
                if 0<=nx2<s and 0<=ny2<s and 0<=nz2<s and self.data[nx2,ny2,nz2]:
                    continue
                sh = .55 + .45*max(0., nrm[0]*.3+nrm[1]*.6+nrm[2]*.1+.25)
                sc = (col[0]*sh, col[1]*sh, col[2]*sh, col[3])
                for i0,i1,i2 in ((0,1,2),(0,2,3)):
                    for vi in (i0,i1,i2):
                        p = fv[vi]
                        V.append((x+p[0], y+p[1], z+p[2]))
                        N.append(nrm); C.append(sc)
        self.face_count   = len(V)//3
        self.vertex_count = len(V)
        if V:
            self.vertices   = np.array(V, dtype=np.float32)
            self.normals    = np.array(N, dtype=np.float32)
            self.colors_arr = np.array(C, dtype=np.float32)
        else:
            self.vertices   = np.zeros((0,3), dtype=np.float32)
            self.normals    = np.zeros((0,3), dtype=np.float32)
            self.colors_arr = np.zeros((0,4), dtype=np.float32)
        self.mesh_dirty = False


# ================================================================
#  Voxel Algorithms
# ================================================================
class VA:
    # ---- generators -------------------------------------------------------
    @staticmethod
    def flat(c, height=1, blk=VoxelChunk.DIRT):
        c.clear()
        for x in range(c.size):
            for z in range(c.size):
                for y in range(min(height, c.size)): c.set(x,y,z,blk)

    @staticmethod
    def heightmap(c, hscale=10, nscale=.05, seed=42):
        c.clear(); pn=PerlinNoise(seed)
        for x in range(c.size):
            for z in range(c.size):
                h = max(1, min(int((pn.octave(x*nscale,0,z*nscale,4)+1)*.5*hscale)+1, c.size-1))
                for y in range(h):
                    c.set(x,y,z, VoxelChunk.GRASS if y==h-1 else (VoxelChunk.DIRT if y>h-4 else VoxelChunk.STONE))

    @staticmethod
    def noise3d(c, thr=0., nscale=.08, seed=42):
        c.clear(); pn=PerlinNoise(seed)
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    n = pn.octave(x*nscale, y*nscale, z*nscale, 3)
                    if n > thr:
                        t = (n-thr)/(1-thr+1e-9)
                        c.set(x,y,z,VoxelChunk.CUSTOM)
                        c.set_color(x,y,z,(.2+.7*t,.35+.45*(1-t),.85-.55*t,1))

    @staticmethod
    def sphere(c, r=None, blk=VoxelChunk.GRASS):
        c.clear(); r=r or c.size//2-1; cx=cy=cz=c.size//2
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    if (x-cx)**2+(y-cy)**2+(z-cz)**2<=r*r: c.set(x,y,z,blk)

    @staticmethod
    def scatter(c, density=.1, seed=42):
        c.clear(); rng=random.Random(seed)
        ts=[VoxelChunk.STONE,VoxelChunk.DIRT,VoxelChunk.GRASS,VoxelChunk.SAND,VoxelChunk.WOOD,VoxelChunk.LEAVES]
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    if rng.random()<density: c.set(x,y,z,rng.choice(ts))

    @staticmethod
    def diamond_square(c, rough=.6, seed=42):
        c.clear(); rng=random.Random(seed); s=c.size
        hm=np.zeros((s,s)); step=s-1
        hm[0,0]=rng.random(); hm[0,step]=rng.random()
        hm[step,0]=rng.random(); hm[step,step]=rng.random()
        sc=rough
        while step>1:
            h=step//2
            for y in range(0,s-1,step):
                for x in range(0,s-1,step):
                    hm[y+h,x+h]=(hm[y,x]+hm[y,x+step]+hm[y+step,x]+hm[y+step,x+step])/4+(rng.random()-.5)*sc
            for y in range(0,s,h):
                for x in range((y+h)%step,s,step):
                    t=n=0
                    if y>=h:   t+=hm[y-h,x]; n+=1
                    if y+h<s:  t+=hm[y+h,x]; n+=1
                    if x>=h:   t+=hm[y,x-h]; n+=1
                    if x+h<s:  t+=hm[y,x+h]; n+=1
                    hm[y,x]=t/n+(rng.random()-.5)*sc
            step=h; sc*=.5
        hm=(hm-hm.min())/(hm.max()-hm.min()+1e-9)
        hs=s//3
        for x in range(s):
            for z in range(s):
                h2=max(1,min(int(hm[x,z]*hs)+1,s-1))
                for y in range(h2):
                    c.set(x,y,z,VoxelChunk.GRASS if y==h2-1 else (VoxelChunk.DIRT if y>h2-3 else VoxelChunk.STONE))

    @staticmethod
    def menger(c, level=2):
        c.clear(); ss=3**level; sc=c.size/ss
        def fill(x,y,z,lv):
            if lv==0: return True
            s2=3**(lv-1)
            mx,my,mz=(x//s2)%3==1,(y//s2)%3==1,(z//s2)%3==1
            if (mx and my)or(mx and mz)or(my and mz): return False
            return fill(x%s2,y%s2,z%s2,lv-1)
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    sx,sy,sz=int(x/sc),int(y/sc),int(z/sc)
                    if sx<ss and sy<ss and sz<ss and fill(sx,sy,sz,level):
                        d=sum(1 for l in range(level) for a in [(sx//(3**l))%3==1,(sy//(3**l))%3==1,(sz//(3**l))%3==1] if a)
                        t=d/max(level*3,1)
                        c.set(x,y,z,VoxelChunk.CUSTOM)
                        c.set_color(x,y,z,(.85-.5*t,.25+.45*t,.82-.3*t,1))

    @staticmethod
    def wave(c, amp=4, freq=.3, blk=VoxelChunk.SAND):
        c.clear(); cy=c.size//2
        for x in range(c.size):
            for z in range(c.size):
                h=max(0,min(int(cy+amp*math.sin(x*freq)*math.cos(z*freq)),c.size-1))
                for y in range(h+1): c.set(x,y,z,blk)

    @staticmethod
    def checker(c, blk=VoxelChunk.STONE):
        c.clear()
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    if (x+y+z)%2==0: c.set(x,y,z,blk)

    @staticmethod
    def worm_cave(c, worms=5, length=60, radius=2, seed=42):
        """Carve worm-like caves through a solid block."""
        c.clear()
        # fill solid
        c.data[:] = VoxelChunk.STONE
        rng = random.Random(seed)
        for _ in range(worms):
            wx = rng.randint(radius, c.size-radius-1)
            wy = rng.randint(c.size//3, 2*c.size//3)
            wz = rng.randint(radius, c.size-radius-1)
            dx, dy, dz = rng.choice([-1,0,1]), rng.choice([-1,0,1]), rng.choice([-1,0,1])
            for _ in range(length):
                for ox in range(-radius, radius+1):
                    for oy in range(-radius, radius+1):
                        for oz in range(-radius, radius+1):
                            if ox*ox+oy*oy+oz*oz <= radius*radius:
                                nx,ny,nz = wx+ox,wy+oy,wz+oz
                                if 0<=nx<c.size and 0<=ny<c.size and 0<=nz<c.size:
                                    c.data[nx,ny,nz] = 0
                dx += rng.choice([-1,0,0,0,1]); dy += rng.choice([-1,0,0,0,1]); dz += rng.choice([-1,0,0,0,1])
                dx = max(-1,min(1,dx)); dy = max(-1,min(1,dy)); dz = max(-1,min(1,dz))
                wx += dx; wy += dy; wz += dz
                wx = max(radius, min(c.size-radius-1, wx))
                wy = max(radius, min(c.size-radius-1, wy))
                wz = max(radius, min(c.size-radius-1, wz))
        c.mesh_dirty = True

    # ---- modifiers --------------------------------------------------------
    @staticmethod
    def erosion(c, iters=5):
        for _ in range(iters):
            ch=[]
            for x in range(1,c.size-1):
                for z in range(1,c.size-1):
                    y=c.size-1
                    while y>0 and c.get(x,y,z)==0: y-=1
                    if y<=0: continue
                    for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny=c.size-1
                        while ny>0 and c.get(x+dx,ny,z+dz)==0: ny-=1
                        if y-ny>1: ch.append((x,y,z,x+dx,ny+1,z+dz))
            for sx,sy,sz,dx,dy,dz in ch:
                b=c.get(sx,sy,sz); c.set(sx,sy,sz,0); c.set(dx,dy,dz,b)

    @staticmethod
    def smooth(c):
        o=c.data.copy()
        for x in range(1,c.size-1):
            for y in range(1,c.size-1):
                for z in range(1,c.size-1):
                    n=int(np.count_nonzero(o[x-1:x+2,y-1:y+2,z-1:z+2]))-int(o[x,y,z]>0)
                    if o[x,y,z]==0 and n>=20: c.set(x,y,z,VoxelChunk.DIRT)
                    elif o[x,y,z]>0 and n<=3: c.set(x,y,z,0)

    @staticmethod
    def flood_fill(c, x, y, z, new_t):
        old=c.get(x,y,z)
        if old==new_t: return
        stk=[(x,y,z)]; vis=set()
        while stk:
            cx,cy,cz=stk.pop()
            if (cx,cy,cz) in vis: continue
            if not(0<=cx<c.size and 0<=cy<c.size and 0<=cz<c.size): continue
            if c.get(cx,cy,cz)!=old: continue
            vis.add((cx,cy,cz)); c.set(cx,cy,cz,new_t)
            stk.extend(((cx+1,cy,cz),(cx-1,cy,cz),(cx,cy+1,cz),(cx,cy-1,cz),(cx,cy,cz+1),(cx,cy,cz-1)))

    @staticmethod
    def add_trees(c, count=10, seed=42):
        rng=random.Random(seed)
        for _ in range(count):
            tx=rng.randint(3,c.size-4); tz=rng.randint(3,c.size-4)
            ty=c.size-1
            while ty>0 and c.get(tx,ty,tz)==0: ty-=1
            if ty<=0: continue
            h=rng.randint(4,7)
            for y in range(h): c.set(tx,ty+1+y,tz,VoxelChunk.WOOD)
            for dx in range(-2,3):
                for dy in range(-1,3):
                    for dz in range(-2,3):
                        if abs(dx)+abs(dy)+abs(dz)<4:
                            nx,ny,nz=tx+dx,ty+1+h+dy,tz+dz
                            if 0<=nx<c.size and 0<=ny<c.size and 0<=nz<c.size and c.get(nx,ny,nz)==0:
                                c.set(nx,ny,nz,VoxelChunk.LEAVES)

    @staticmethod
    def invert(c):
        m=c.data==0; c.data[m]=VoxelChunk.STONE; c.data[~m]=0; c.mesh_dirty=True

    @staticmethod
    def subsample(c, f=2):
        ns=max(4,c.size//f); nc=VoxelChunk(ns)
        for x in range(ns):
            for y in range(ns):
                for z in range(ns): nc.data[x,y,z]=c.data[x*f,y*f,z*f]
        return nc

    # ---- color modifiers --------------------------------------------------
    @staticmethod
    def color_height(c):
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    if c.data[x,y,z]:
                        t=y/max(c.size-1,1)
                        c.set_color(x,y,z,(.1+.9*t,.82-.6*t,.2+.3*(1-t),1))

    @staticmethod
    def color_noise(c, seed=42):
        pn=PerlinNoise(seed)
        for x in range(c.size):
            for y in range(c.size):
                for z in range(c.size):
                    if c.data[x,y,z]:
                        n=(pn.noise3d(x*.12,y*.12,z*.12)+1)*.5
                        c.set_color(x,y,z,(.3+.7*n,.2+.5*(1-n),.3+.4*abs(n-.5)*2,1))

    # ---- CSG --------------------------------------------------------------
    @staticmethod
    def csg_union(a,b):
        r=VoxelChunk(a.size); r.data=np.maximum(a.data,b.data); r.colors={**b.colors,**a.colors}; r.mesh_dirty=True; return r

    @staticmethod
    def csg_intersect(a,b):
        r=VoxelChunk(a.size); m=(a.data>0)&(b.data>0)
        r.data=np.where(m,a.data,0).astype(np.uint8)
        r.colors={k:v for k,v in a.colors.items() if m[k]}; r.mesh_dirty=True; return r

    @staticmethod
    def csg_diff(a,b):
        r=VoxelChunk(a.size); m=b.data==0
        r.data=np.where(m,a.data,0).astype(np.uint8)
        r.colors={k:v for k,v in a.colors.items() if m[k]}; r.mesh_dirty=True; return r

    @staticmethod
    def make_shape(size, shape):
        c=VoxelChunk(size); cx=cy=cz=size//2; r=size//4
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    dx,dy,dz=x-cx,y-cy,z-cz; ok=False
                    if shape=="Sphere":   ok=dx*dx+dy*dy+dz*dz<=r*r
                    elif shape=="Cube":   ok=abs(dx)<=r and abs(dy)<=r and abs(dz)<=r
                    elif shape=="Cylinder":ok=dx*dx+dz*dz<=r*r and abs(dy)<=r
                    if ok: c.set(x,y,z,VoxelChunk.SAND)
        return c


# ================================================================
#  OpenGL Viewport
# ================================================================
class VoxelViewport(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chunk = VoxelChunk(32)
        self.cam_dist=50.; self.cam_rx=30.; self.cam_ry=45.
        self.cam_tgt=[16.,8.,16.]
        self._lp=None; self._mb=None
        self.wireframe=False; self.show_grid=True; self.show_axes=True; self.show_bounds=True
        self.fps=0.; self._fc=0; self._ft=time.time()
        self.setMinimumSize(400,300); self.setFocusPolicy(Qt.StrongFocus)

    def initializeGL(self):
        glClearColor(.12,.12,.16,1)
        glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0,GL_POSITION,[1,2,1,0])
        glLightfv(GL_LIGHT0,GL_AMBIENT,[.3,.3,.3,1])
        glLightfv(GL_LIGHT0,GL_DIFFUSE,[.8,.8,.8,1])
        glLightfv(GL_LIGHT0,GL_SPECULAR,[.2,.2,.2,1])
        glEnable(GL_COLOR_MATERIAL); glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)

    def resizeGL(self, w, h):
        glViewport(0,0,w,h); glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45, w/max(h,1), .1, 500); glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT); glLoadIdentity()
        rx,ry=math.radians(self.cam_rx),math.radians(self.cam_ry)
        ex=self.cam_tgt[0]+self.cam_dist*math.cos(rx)*math.sin(ry)
        ey=self.cam_tgt[1]+self.cam_dist*math.sin(rx)
        ez=self.cam_tgt[2]+self.cam_dist*math.cos(rx)*math.cos(ry)
        gluLookAt(ex,ey,ez,*self.cam_tgt,0,1,0)
        glLightfv(GL_LIGHT0,GL_POSITION,[ex+10,ey+20,ez+10,0])
        if self.show_grid:   self._grid()
        if self.show_axes:   self._axes()
        if self.show_bounds: self._bounds()
        self._voxels()
        self._fc+=1; now=time.time(); dt=now-self._ft
        if dt>=1.: self.fps=self._fc/dt; self._fc=0; self._ft=now

    def _grid(self):
        glDisable(GL_LIGHTING); glColor4f(.35,.35,.35,.5); glBegin(GL_LINES)
        s=self.chunk.size
        for i in range(s+1):
            glVertex3f(i,0,0); glVertex3f(i,0,s)
            glVertex3f(0,0,i); glVertex3f(s,0,i)
        glEnd(); glEnable(GL_LIGHTING)

    def _axes(self):
        glDisable(GL_LIGHTING); glLineWidth(2.5); glBegin(GL_LINES)
        for c,v in [((1,.2,.2),(6,0,0)),((.2,1,.2),(0,6,0)),((.3,.3,1),(0,0,6))]:
            glColor3f(*c); glVertex3f(0,0,0); glVertex3f(*v)
        glEnd(); glLineWidth(1); glEnable(GL_LIGHTING)

    def _bounds(self):
        glDisable(GL_LIGHTING); glColor4f(.55,.55,.75,.25); s=self.chunk.size
        glBegin(GL_LINES)
        for x in(0,s):
            for y in(0,s): glVertex3f(x,y,0); glVertex3f(x,y,s)
        for x in(0,s):
            for z in(0,s): glVertex3f(x,0,z); glVertex3f(x,s,z)
        for y in(0,s):
            for z in(0,s): glVertex3f(0,y,z); glVertex3f(s,y,z)
        glEnd(); glEnable(GL_LIGHTING)

    def _voxels(self):
        if self.chunk.mesh_dirty: self.chunk.build_mesh()
        if self.chunk.vertex_count==0: return
        if self.wireframe: glPolygonMode(GL_FRONT_AND_BACK,GL_LINE); glDisable(GL_LIGHTING)
        glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY); glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3,GL_FLOAT,0,self.chunk.vertices)
        glNormalPointer(GL_FLOAT,0,self.chunk.normals)
        glColorPointer(4,GL_FLOAT,0,self.chunk.colors_arr)
        glDrawArrays(GL_TRIANGLES,0,self.chunk.vertex_count)
        glDisableClientState(GL_VERTEX_ARRAY); glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_COLOR_ARRAY)
        if self.wireframe: glPolygonMode(GL_FRONT_AND_BACK,GL_FILL); glEnable(GL_LIGHTING)

    def mousePressEvent(self,e):   self._lp=e.position(); self._mb=e.button()
    def mouseReleaseEvent(self,e): self._lp=None; self._mb=None

    def mouseMoveEvent(self,e):
        if self._lp is None: return
        dx=e.position().x()-self._lp.x(); dy=e.position().y()-self._lp.y()
        if self._mb==Qt.LeftButton:
            self.cam_ry+=dx*.5; self.cam_rx=max(-89,min(89,self.cam_rx+dy*.5))
        elif self._mb==Qt.RightButton:
            self.cam_tgt[0]-=dx*.05; self.cam_tgt[1]+=dy*.05
        elif self._mb==Qt.MiddleButton:
            self.cam_dist=max(5,min(200,self.cam_dist+dy*.15))
        self._lp=e.position(); self.update()

    def wheelEvent(self,e):
        self.cam_dist=max(5,min(200,self.cam_dist-e.angleDelta().y()/120*2)); self.update()


# ================================================================
#  Main Window
# ================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voxel Engine Testing Tool")
        self.resize(1300, 840)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        self.viewport = VoxelViewport()
        splitter.addWidget(self.viewport)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(295); scroll.setMaximumWidth(370)
        panel = QWidget(); lay = QVBoxLayout(panel); scroll.setWidget(panel)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0,3); splitter.setStretchFactor(1,0)

        # ===== GENERATION ==================================================
        gb = QGroupBox("🧊 Generation Algorithms"); gl = QVBoxLayout(gb)
        self.gen_combo = QComboBox()
        self.gen_combo.addItems([
            "Flat Terrain","Perlin Heightmap","3D Perlin Noise","Sphere",
            "Random Scatter","Diamond-Square","Menger Sponge",
            "Sine Wave","Checkerboard","Worm Cave        "])
        gl.addWidget(self.gen_combo)

        row = QHBoxLayout(); row.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0,99999); self.seed_spin.setValue(42)
        row.addWidget(self.seed_spin); gl.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("Chunk Size:"))
        self.size_combo = QComboBox(); self.size_combo.addItems(["16","32","48","64"])
        self.size_combo.setCurrentIndex(1); row.addWidget(self.size_combo); gl.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("Height Scale:"))
        self.hscale = QSpinBox(); self.hscale.setRange(1,40); self.hscale.setValue(10)
        row.addWidget(self.hscale); gl.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("Noise Scale:"))
        self.nscale = QDoubleSpinBox(); self.nscale.setRange(0.01,0.5); self.nscale.setSingleStep(0.01); self.nscale.setValue(0.05)
        row.addWidget(self.nscale); gl.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("Threshold:"))
        self.thresh = QDoubleSpinBox(); self.thresh.setRange(-1,1); self.thresh.setSingleStep(0.05); self.thresh.setValue(0.0)
        row.addWidget(self.thresh); gl.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("Density:"))
        self.density = QDoubleSpinBox(); self.density.setRange(0.01,0.5); self.density.setSingleStep(0.01); self.density.setValue(0.1)
        row.addWidget(self.density); gl.addLayout(row)

        btn = QPushButton("⚡ Generate")
        btn.setStyleSheet("QPushButton{background:#4CAF50;color:#fff;font-weight:bold;padding:7px;border-radius:4px;}")
        btn.clicked.connect(self._on_generate)
        gl.addWidget(btn)
        lay.addWidget(gb)

        # ===== OPERATIONS ===================================================
        gb = QGroupBox("🔧 Voxel Operations"); ol = QVBoxLayout(gb)
        self.ops_combo = QComboBox()
        self.ops_combo.addItems([
            "Erosion", "Smooth", "Flood Fill (Center → Grass)",
            "Color by Height", "Color by Noise",
            "Add Trees", "Invert", "Subsample (2×)", "Clear Top Layer",
        ])
        ol.addWidget(self.ops_combo)

        row = QHBoxLayout(); row.addWidget(QLabel("Iterations:"))
        self.iter_spin = QSpinBox(); self.iter_spin.setRange(1,50); self.iter_spin.setValue(5)
        row.addWidget(self.iter_spin); ol.addLayout(row)

        btn = QPushButton("⚙ Apply Operation")
        btn.setStyleSheet("QPushButton{background:#2196F3;color:#fff;font-weight:bold;padding:7px;border-radius:4px;}")
        btn.clicked.connect(self._on_operation)
        ol.addWidget(btn)
        lay.addWidget(gb)

        # ===== CSG ==========================================================
        gb = QGroupBox("✂ CSG Boolean"); cl = QVBoxLayout(gb)
        cl.addWidget(QLabel("Combine current voxels with shape:"))
        self.csg_shape = QComboBox(); self.csg_shape.addItems(["Sphere","Cube","Cylinder"])
        cl.addWidget(self.csg_shape)
        row = QHBoxLayout()
        for name, op in [("Union","union"),("Intersect","intersect"),("Subtract","diff")]:
            b = QPushButton(name)
            b.clicked.connect(lambda _, o=op: self._on_csg(o))
            row.addWidget(b)
        cl.addLayout(row)
        lay.addWidget(gb)

        # ===== VISUALIZATION ================================================
        gb = QGroupBox("👁 Visualization"); vl = QVBoxLayout(gb)
        for label, attr, default in [
            ("Wireframe","wireframe",False), ("Show Grid","show_grid",True),
            ("Show Axes","show_axes",True), ("Show Bounds","show_bounds",True),
        ]:
            cb = QCheckBox(label); cb.setChecked(default)
            cb.toggled.connect(lambda v, a=attr: (setattr(self.viewport, a, v), self.viewport.update()))
            vl.addWidget(cb)
        lay.addWidget(gb)

        # ===== CAMERA =======================================================
        gb = QGroupBox("📷 Camera"); cal = QVBoxLayout(gb)
        for label, rx, ry in [("Reset",30,45),("Top",89,0),("Front",0,0),("Side",0,90),("Iso",30,45)]:
            b = QPushButton(label)
            b.clicked.connect(lambda _, x=rx, y=ry: self._set_cam(x, y))
            cal.addWidget(b)
        lay.addWidget(gb)

        # ===== STATISTICS ===================================================
        gb = QGroupBox("📊 Statistics"); sl = QVBoxLayout(gb)
        self.stats_lbl = QLabel("Voxels: 0\nFaces: 0\nVertices: 0\nFPS: --")
        self.stats_lbl.setStyleSheet("font-family:monospace;font-size:12px;")
        sl.addWidget(self.stats_lbl)
        lay.addWidget(gb)

        # ===== UTILITY ======================================================
        gb = QGroupBox("🛠 Utility"); ul = QVBoxLayout(gb)
        btn = QPushButton("🗑 Clear All")
        btn.setStyleSheet("QPushButton{background:#f44336;color:#fff;font-weight:bold;padding:7px;border-radius:4px;}")
        btn.clicked.connect(self._on_clear)
        ul.addWidget(btn)
        btn = QPushButton("⏱ Benchmark")
        btn.setStyleSheet("QPushButton{background:#FF9800;color:#fff;font-weight:bold;padding:7px;border-radius:4px;}")
        btn.clicked.connect(self._on_benchmark)
        ul.addWidget(btn)
        lay.addWidget(gb)

        lay.addStretch()

        # --- timers & status ------------------------------------------------
        self.statusBar().showMessage("LMB = Orbit | RMB = Pan | Wheel = Zoom")
        self._stats_timer = QTimer(); self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(400)
        QTimer.singleShot(150, self._on_generate)

    # ---- helpers -----------------------------------------------------------
    def _chunk_size(self): return int(self.size_combo.currentText())

    def _set_cam(self, rx, ry):
        s = self.viewport.chunk.size
        self.viewport.cam_rx = rx; self.viewport.cam_ry = ry
        self.viewport.cam_tgt = [s/2, s/4, s/2]
        self.viewport.cam_dist = s * 1.6
        self.viewport.update()

    # ---- slots -------------------------------------------------------------
    def _on_generate(self):
        sz = self._chunk_size(); c = VoxelChunk(sz); seed = self.seed_spin.value()
        algo = self.gen_combo.currentText()
        t0 = time.time()
        if   algo == "Flat Terrain":      VA.flat(c, height=self.hscale.value())
        elif algo == "Perlin Heightmap":  VA.heightmap(c, hscale=self.hscale.value(), nscale=self.nscale.value(), seed=seed)
        elif algo == "3D Perlin Noise":   VA.noise3d(c, thr=self.thresh.value(), nscale=self.nscale.value(), seed=seed)
        elif algo == "Sphere":            VA.sphere(c, radius=sz//2-1)
        elif algo == "Random Scatter":    VA.scatter(c, density=self.density.value(), seed=seed)
        elif algo == "Diamond-Square":    VA.diamond_square(c, rough=.6, seed=seed)
        elif algo == "Menger Sponge":     VA.menger(c, level=2 if sz >= 27 else 1)
        elif algo == "Sine Wave":         VA.wave(c, amp=self.hscale.value())
        elif algo == "Checkerboard":      VA.checker(c)
        elif algo == "Worm Cave":         VA.worm_cave(c, worms=5, length=60, radius=2, seed=seed)
        ms = (time.time()-t0)*1000
        self.viewport.chunk = c; self._set_cam(30, 45); self.viewport.update()
        self.statusBar().showMessage(f"Generated '{algo}' in {ms:.1f} ms  (size={sz})")

    def _on_operation(self):
        op = self.ops_combo.currentText(); c = self.viewport.chunk; t0 = time.time()
        if   op == "Erosion":                   VA.erosion(c, iters=self.iter_spin.value())
        elif op == "Smooth":                    VA.smooth(c)
        elif op == "Flood Fill (Center → Grass)":
            cx, cz = c.size//2, c.size//2; cy = 0
            while cy < c.size and c.get(cx, cy, cz) == 0: cy += 1
            if cy < c.size: VA.flood_fill(c, cx, cy, cz, VoxelChunk.GRASS)
        elif op == "Color by Height":           VA.color_height(c)
        elif op == "Color by Noise":            VA.color_noise(c, seed=self.seed_spin.value())
        elif op == "Add Trees":                 VA.add_trees(c, count=10, seed=self.seed_spin.value())
        elif op == "Invert":                    VA.invert(c)
        elif op == "Subsample (2×)":
            nc = VA.subsample(c, 2); self.viewport.chunk = nc; c = nc
        elif op == "Clear Top Layer":
            for y in range(c.size-1, -1, -1):
                found = False
                for x in range(c.size):
                    for z in range(c.size):
                        if c.data[x,y,z] > 0: c.set(x,y,z,0); found = True
                if found: break
        ms = (time.time()-t0)*1000; self.viewport.update()
        self.statusBar().showMessage(f"Applied '{op}' in {ms:.1f} ms")

    def _on_csg(self, op):
        c = self.viewport.chunk; b = VA.make_shape(c.size, self.csg_shape.currentText())
        t0 = time.time()
        if   op == "union":     r = VA.csg_union(c, b)
        elif op == "intersect": r = VA.csg_intersect(c, b)
        elif op == "diff":      r = VA.csg_diff(c, b)
        ms = (time.time()-t0)*1000
        self.viewport.chunk = r; self.viewport.update()
        self.statusBar().showMessage(f"CSG {op} in {ms:.1f} ms")

    def _on_clear(self):
        self.viewport.chunk = VoxelChunk(self._chunk_size())
        self.viewport.update(); self.statusBar().showMessage("Cleared")

    def _on_benchmark(self):
        lines = []
        for sz in (16, 32):
            c = VoxelChunk(sz)
            t0 = time.time(); VA.heightmap(c, hscale=sz//3, seed=42)
            g = (time.time()-t0)*1000
            t0 = time.time(); c.build_mesh()
            m = (time.time()-t0)*1000
            lines.append(f"Terrain {sz}³: gen {g:.1f}ms | mesh {m:.1f}ms | vox {c.count_voxels():,} | tri {c.face_count:,}")
        c = VoxelChunk(16)
        t0 = time.time(); VA.noise3d(c, thr=0., seed=42)
        g = (time.time()-t0)*1000
        t0 = time.time(); c.build_mesh()
        m = (time.time()-t0)*1000
        lines.append(f"3D Noise 16³: gen {g:.1f}ms | mesh {m:.1f}ms | vox {c.count_voxels():,}")
        QMessageBox.information(self, "Benchmark", "\n".join(lines))

    def _update_stats(self):
        c = self.viewport.chunk
        self.stats_lbl.setText(
            f"Voxels:   {c.count_voxels():>8,}\n"
            f"Faces:    {c.face_count:>8,}\n"
            f"Vertices: {c.vertex_count:>8,}\n"
            f"FPS:      {self.viewport.fps:>8.1f}"
        )


# ================================================================
#  Entry Point
# ================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    p = QPalette()
    p.setColor(QPalette.Window,          QColor(53, 53, 53))
    p.setColor(QPalette.WindowText,      QColor(255,255,255))
    p.setColor(QPalette.Base,            QColor(35, 35, 35))
    p.setColor(QPalette.AlternateBase,   QColor(53, 53, 53))
    p.setColor(QPalette.ToolTipBase,     QColor(25, 25, 25))
    p.setColor(QPalette.ToolTipText,     QColor(255,255,255))
    p.setColor(QPalette.Text,            QColor(255,255,255))
    p.setColor(QPalette.Button,          QColor(53, 53, 53))
    p.setColor(QPalette.ButtonText,      QColor(255,255,255))
    p.setColor(QPalette.BrightText,      QColor(255, 50, 50))
    p.setColor(QPalette.Link,            QColor(42, 130,218))
    p.setColor(QPalette.Highlight,       QColor(42, 130,218))
    p.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
    app.setPalette(p)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()