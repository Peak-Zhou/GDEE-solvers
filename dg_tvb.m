function u=dg_tvb(dt,h,N,M,ut0,xdot,E_array,a_vals)
u=zeros(M,N);
c=zeros(3,M,N);
for i= 1:M
    c(:,i,1)=[ut0(i);0;0];
end
for n=1:N-1
    a=xdot(n);
    c(:,:,n+1)=RK4_fast(c(:,:,n),dt,h,a,M,E_array,a_vals);
    c(:,:,n+1)=Limiter_fast(c(:,:,n+1),M,h);
    c(:,1,n+1)=c(:,1,n);
    c(:,M,n+1)=c(:,M,n);
end
for n=1:N
    for j=1:M
        u(j,n)=[1,0,-1/2]*c(:,j,n);
    end
end
end

function c_new=RK4_fast(c0,dt,h,a,M,E_array,a_vals)
    c1=c0;
    c2=c0;
    for i=1:5
        c1=c1+(dt/6)*Lh_fast(c1,h,a,M,E_array,a_vals);
    end
    c2=(1/25)*c2+(9/25)*c1;
    c1=15*c2-5*c1;
    for i=6:9
        c1=c1+(dt/6)*Lh_fast(c1,h,a,M,E_array,a_vals);
    end
    c_new=c2+(3/5)*c1+(dt/10)*Lh_fast(c1,h,a,M,E_array,a_vals);
end

function Lh=Lh_fast(c0,h,a,M,E_array,a_vals)
    c0_vec=reshape(c0,3*M,1);
    idx=nearest_a(a,a_vals);
    E=E_array(:,:,idx);
    Lh_vec=-E*c0_vec;
    Lh=reshape(Lh_vec,3,M);
end

function idx=nearest_a(a,a_vals)
    [~, idx]=min(abs(a_vals-a));
end

function c=Limiter_fast(c,M,h)
    Q=[1,0,0;1,1,1;1,-1,1];
    MM=500000;
    for j=2:M-1
        v=c(:,j);
        ubar=Q(1,:)*v;
        utilde=Q(2,:)*v-ubar;
        uhat=ubar-Q(3,:)*v;
        DeltaR=Q(1,:)*c(:,j+1)-Q(1,:)*v;
        DeltaL=Q(1,:)*v-Q(1,:)*c(:,j-1);
        X1=[utilde,DeltaR,DeltaL];
        X2=[uhat,DeltaR,DeltaL];
        utildemod=ubar+minmod2(X1,MM,h);
        uhatmod=ubar-minmod2(X2,MM,h);
        b=[ubar;utildemod;uhatmod];
        c(:,j)=Q\b;
    end
end

function m=minmod2(X,M,h)
    if abs(X(1))<=M*h^2
        m=X(1);
    else
        m=minmod(X);
    end
end

function m=minmod(X)
    s=sum(sign(X))/length(X);
    if abs(s)==1
        m=s*min(abs(X));
    else
        m=0;
    end
end