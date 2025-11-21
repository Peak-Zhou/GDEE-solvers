function val=intpoly(f,x,y)
pos=[x,y];
X=pos(:,1);Y=pos(:,2);
k=convhull(X,Y);
X=X(k);Y=Y(k);
xmin=min(X);xmax=max(X);
ymin=min(Y);ymax=max(Y);
g=@(x,y)f(x,y).*inpolygon(x,repmat(y,size(x)),X,Y);
val=quadl(@(y)arrayfun(@(y)quadl(@(x)g(x,y),xmin,xmax),y),ymin,ymax);
end
