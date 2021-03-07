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

; ------------------------------------------------
; npx is the x size of box; npy is y size of box ; step1 is the first several step size, while the stepf is the final one;
; nnx is the box number in x direction; nny is that on y direction; 
; idlist will output a corresponding x,y box indice, and corresponding pixel postion
function padsystem_2d,data=data,npx=npx,npy=npy,step1=step1,stepf=stepf,nnx=nnx,nny=nny,idlist=idlist
nx = n_elements(data[*,0,0])
ny = n_elements(data[0,*,*])
result=reform(data[0:(npx-1),0:(npy-1),*],npx*npy*41.)
idlist = fltarr(5,nnx*nny)
k = 1
for j = 0,nny-1 do begin
  for i = 0,nnx-1 do begin
    if j eq 0 and i eq 0 then continue
    if i eq nnx-1 then xx = stepf else xx = step1 
    if j eq nny-1 then yy = stepf else yy = step1 
    temp=reform(data[((i-1)*step1+xx):((i-1)*step1+xx+npx-1),((j-1)*step1+yy):((j-1)*step1+yy+npy-1),*],npx*npy*41.)
    result = [[result],[temp]]
    idlist(0,k) = i
    idlist(1,k) = j
    idlist(2,k) = ((i-1)*step1+xx)
    idlist(3,k) = ((j-1)*step1+yy)
    idlist(4,k) = 0.
    k=k+1
  endfor
endfor

return,result
end 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

