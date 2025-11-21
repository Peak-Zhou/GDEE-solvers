function u=fv_weno(dt,h,N,M,ut0,xdot)
u=zeros(M,N);
u(:,1)=ut0;
for n=1:N-1
    a=xdot(n);
    u(:,n+1)=RK4(u(:,n),dt,h,a);
    u(1,n+1)=u(1,n);
    u(M,n+1)=u(M,n);
end

function u=RK4(u0,dt,h,a)
u1=u0;
u2=u0;
for i=1:5
    u1=u1+(dt/6)*Lh(u1,h,a);
end
u2=(1/25)*u2+(9/25)*u1;
u1=15*u2-5*u1;
for i=6:9
    u1=u1+(dt/6)*Lh(u1,h,a);
end
u=u2+(3/5)*u1+(dt/10)*Lh(u1,h,a);

function Lh=Lh(u0,h,a)
f=@(x)a*x;
M=length(u0);
uL=zeros(M+1,1);
uR=zeros(M+1,1);
uL(1)=11/6*u0(1)-7/6*u0(2)+1/3*u0(3);
uL(2)=1/3*u0(1)+5/6*u0(2)-1/6*u0(3);
uL(3)=-1/6*u0(1)+5/6*u0(2)+1/3*u0(3);
uL(M)=-1/6*u0(M-2)+5/6*u0(M-1)+1/3*u0(M);
uL(M+1)=1/3*u0(M-2)-7/6*u0(M-1)+11/6*u0(M);
uR(1)=uL(1);
uR(2)=uL(2);
uR(M-1)=-1/6*u0(M-3)+5/6*u0(M-2)+1/3*u0(M-1);
uR(M)=uL(M);
uR(M+1)=uL(M+1);
for i=4:M-1
    uL(i)=WENO(u0(i-3:i+1));
end
for i=3:M-2
    uR(i)=WENO(u0(i+2:-1:i-2));
end
fhat=zeros(M+1,1);
for i=1:M+1
    fhat(i)=1/2*(f(uR(i))+f(uL(i))-abs(a)*(uR(i)-uL(i)));
end
Lh=-(fhat(2:end)-fhat(1:end-1))/h;

function uu=WENO(uRL)
epsilon=1e-8;
LL=uRL(1);
L=uRL(2);
C=uRL(3);
R=uRL(4);
RR=uRL(5);
U2=1/3*LL-7/6*L+11/6*C;
U1=-1/6*L+5/6*C+1/3*R;
U0=1/3*C+5/6*R-1/6*RR;
beta2=13/12*(LL-2*L+C)^2+1/4*(LL-4*L+3*C)^2;
beta1=13/12*(L-2*C+R)^2+1/4*(L-R)^2;
beta0=13/12*(C-2*R+RR)^2+1/4*(3*C-4*R+RR)^2;
tomega2=1/(epsilon+beta2)^2;
tomega1=6/(epsilon+beta1)^2;
tomega0=3/(epsilon+beta0)^2;
S=tomega0+tomega1+tomega2;
omega2=tomega2/S;
omega1=tomega1/S;
omega0=tomega0/S;
uu=omega0*U0+omega1*U1+omega2*U2;