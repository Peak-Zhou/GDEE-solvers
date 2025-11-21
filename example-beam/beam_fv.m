clc;
clear;
close all;

%system parameters
L=9;                       % (m)
m=250;                     % (kg/m)
EI=4e7;                    % (N`m2)
w1=pi^2*sqrt(EI/(m*L^4));  % (rad/s)

w=0.6*w1;
A1=4e4;
A2=6e4;
phi1=pi/4;
phi2=3*pi/4;

%representative points 
s=2;
n=34;
point=h_w(s,n,[1,21]);
A=A1+(A2-A1)*point(:,1);
phi=phi1+(phi2-phi1)*point(:,2);

Pq=zeros(n,1);
B=[0 0;0 1;1 1;1 0];
pbound=Polyhedron(B);
V=mpt_voronoi([point(:,1),point(:,2)]','bound',pbound);
for iPoly=1:n
    k=convhull(V.Set(iPoly).V(:,1),V.Set(iPoly).V(:,2));
    [pX,pY]=poly2cw(V.Set(iPoly).V(k,1),V.Set(iPoly).V(k,2));
    pX=pX(1:end-1,1);
    pY=pY(1:end-1,1);
    irPoly{iPoly}=[pX,pY];
end
fun=@(x,y)1./(x.*y).*(x.*y);
for p=1:n
   Pq(p)=intpoly(fun,irPoly{p}(:,1),irPoly{p}(:,2));
end
Pq=Pq/sum(Pq);

%physical solutions
N=round(1200*2.5*1);
t=linspace(0,1.2,N+1);
dt=t(2)-t(1);
T=ones(1,length(t));
ydot=2*A*T/(m*L).*(w*sin(phi)*cos(w1*t)-w*sin(w*ones(n,1)*t+phi*T)+w1*cos(phi)*sin(w1*t))./(w1^2-w.^2*T);

%probability solutions
y0=0.1;
M=round(400*1);
y=linspace(-2*y0,2*y0,M+1);
h=y(2)-y(1);

p=zeros(M+1,N+1);
parfor i=1:n
    p0=zeros(M+1,1);
    p0(M/2+1)=Pq(i)/h;
    pyt=fv_weno(dt,h,N+1,M+1,p0,ydot(i,:));
    p=p+pyt;
    i
end

tfv=t;
yfv=y;
pfv=p;
save beam_exc_fv.mat tfv yfv pfv 