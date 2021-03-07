function box_pixel1,cp=cp,bg=bg,xrange=xrange,yrange=yrange,zrange=zrange,rdshift=rdshift
  
  xc=round(cp(0,*))
  yc=round(cp(1,*))
  zc=round(cp(2,*))
  
  
  if keyword_set(rdshift) then begin
    shiftx = fix(randomu(undefinevar,1)*2*rdshift) - rdshift
    shifty = fix(randomu(undefinevar,1)*2*rdshift) - rdshift 
    xc = xc + shiftx
    yc = yc + shifty
  endif  
  temp=bg[(xc-xrange):(xc+xrange),(yc-yrange):(yc+yrange),(zc-zrange):(zc+zrange)]
  return,temp
end

;---------------------------------------------------------------------------------
function flatten3d,data=data  
  
  nz=n_elements(data(0,0,*))
  ny=n_elements(data(0,*,0))
  result=[-1]
  for i = 0,nz-1 do begin
     for j = 0,ny-1 do begin
        result=[result,data(*,j,i)]
     endfor
  endfor
  n=n_elements(result)
  return,result(1:n-1)
  
end

;--------------------------------------------------------------------------
function pixel_boxall,cp=cp,bg=bg,xrange=xrange,yrange=yrange,zrange=zrange,rdshift=rdshift
  n=n_elements(cp(0,*))
  result=flatten3d(data=box_pixel1(cp=cp(*,0),bg=bg,xrange=xrange,yrange=yrange,zrange=zrange,rdshift=rdshift))
  
  for i = 1. ,n-1 do begin
     temp=flatten3d(data=box_pixel1(cp=cp(*,i),bg=bg,xrange=xrange,yrange=yrange,zrange=zrange,rdshift=rdshift))  /84
     result=[[result],[temp]]
  endfor
  
  return,result
end 
  
