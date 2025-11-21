function u=rbf(dt,h,N,M,ut0,xdot,x)
u=zeros(M,N);
u(:,1)=ut0;
c=3*h;
A = zeros(M, M);
B = zeros(M, M);
for i = 1:M
    A(:,i) = sqrt((x(1:M) - x(i)).^2 + c^2);
    B(:,i) = (x(1:M) - x(i)) ./ sqrt((x(1:M) - x(i)).^2 + c^2);
end
[L,U,P]=lu(A);
C=U\(L\(P*B));
lam=U\(L\(P*u(:,1)));
for n=1:N-1
    a=xdot(n);
    lam=RK4(lam,dt,a,C);
    u(:,n+1)=A*lam;
    u(1,n+1)=0;
    u(M,n+1)=0;
    lam=U\(L\(P*u(:,n+1)));
end

function lam=RK4(lam0,dt,a,C)
lam1=lam0;
lam2=lam0;
for i=1:5
    lam1=lam1+(dt/6)*Lh(lam1,a,C);
end
lam2=(1/25)*lam2+(9/25)*lam1;
lam1=15*lam2-5*lam1;
for i=6:9
    lam1=lam1+(dt/6)*Lh(lam1,a,C);
end
lam=lam2+(3/5)*lam1+(dt/10)*Lh(lam1,a,C);

function Lh=Lh(lam0,a,C)
Lh=-a*C*lam0;