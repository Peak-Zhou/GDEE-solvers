function u=rkp(dt,h,N,M,m,xi,xj,ut0,xdot)
u=zeros(M, N);
% u(:,1)=ut0;
H=@(x,y)1;
W=@(R)(2/3-R.^2+R.^3/2).*(R>=0&R<1)+((2-R).^3/6).*(R>=1&R<2);
[Xi,Xj]=ndgrid(xi,xj);
D=abs(Xi-Xj)/(3*h);
Wmat=W(D)*(2*h);
Hmat=ones(M,m);
Mbold=sum(Hmat.*Hmat.*Wmat,2);
G=(H(0,0)./Mbold).*Hmat.*Wmat;
Gprime=zeros(M,m);
Gprime(1,:)=(G(2,:)-G(1,:))/h;
Gprime(2:M-1,:)=(G(3:M,:)-G(1:M-2,:))/(2*h);
Gprime(M,:)=(G(M,:)-G(M-1,:))/h;
U=ut0;
for n=1:N-1
    a=xdot(n);
    U=RK4(U,dt,a,G,Gprime);
    u(:,n+1)=G*U;
    u(1,n+1)=0;
    u(M,n+1)=0;
    U=G\u(:,n+1);
end

end

function U=RK4(U0,dt,a,G,Gprime)
U1=U0;
U2=U0;
for i=1:5
    U1=U1+(dt/6)*Lh(U1,a,G,Gprime);
end
U2=(1/25)*U2+(9/25)*U1;
U1=15*U2-5*U1;
for i=6:9
    U1=U1+(dt/6)*Lh(U1,a,G,Gprime);
end
U=U2+(3/5)*U1+(dt/10)*Lh(U1,a,G,Gprime);

end

function Lh=Lh(U0,a,G,Gprime)
Lh=-a*(G\(Gprime*U0));
end
