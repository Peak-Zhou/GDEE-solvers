function x=h_w(s,n,Q)
x=zeros(n,s);
for i=1:n
    for j=1:s
        h=(2*i*Q(j)-1)/(2*n);
        x(i,j)=h-round(h)+0.5;
    end
end