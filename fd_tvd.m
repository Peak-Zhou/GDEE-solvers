function u=fd_tvd(dt,h,N,M,u0,xdot)
u=zeros(M,N);
u(:,1)=u0;
lam=dt/h;
for n=1:N-1
    a=xdot(n);
    u(2,n+1)=(1-lam^2*a^2)*u(2,n)+1/2*(lam^2*a^2-lam*a)*u(3,n)+1/2*(lam^2*a^2+lam*a)*u(1,n);
    for j=2:M-3
        u(j+1,n+1)=superbee(u(j-1:j+3,n),lam,a);
    end
    u(M-1,n+1)=(1-lam^2*a^2)*u(M-1,n)+1/2*(lam^2*a^2-lam*a)*u(M,n)+1/2*(lam^2*a^2+lam*a)*u(M-2,n);
end

function u=superbee(uRL,lam,a)
du=uRL(2:5)-uRL(1:4);
r=[du(1)/du(2),du(3)/du(2),du(2)/du(3),du(4)/du(3)];
phi=zeros(1,4);
for i=1:4
    phi(i)=max([0,min(2*r(i),1),min(r(i),2)]);
end
phi1=phi(1)*(a>0)+phi(2)*(-a>0);
phi2=phi(3)*(a>0)+phi(4)*(-a>0);
u=uRL(3)-1/2*(lam*a-abs(lam*a))*du(3)-1/2*(lam*a+abs(lam*a))*du(2)-1/2*(abs(lam*a)-lam^2*a^2)*(phi2*du(3)-phi1*du(2));