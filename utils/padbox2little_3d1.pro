; ------------------------------------------------
; npx is the x size of box; npy is y size of box ; step1 is the first several step size, while the stepf is the final one;
; nnx is the box number in x direction; nny is that on y direction; 
; idlist will output a corresponding x,y box indice
; lboxsize is the size of large box, like [61,61,41]
; lboxpos is the initial pixel positon of the origin large box, containing xyz , like [125,75,43]
; sboxsize is the target small box size, like [31,31,21]
; stepxy, like 30
; stepz, like 20
; nnxy is the number of box on xy direction
; nnz is the number of box on z direction
; pad the bigbox into little ones
; 
function padbox2little_3d1,data=data,lboxsize,lboxpos,sboxsize,stepxy=stepxy,stepz=stepz,nnxy=nnxy,nnz=nnz,poslist = poslist

nx = lboxsize(0)
ny = lboxsize(1)
nz = lboxsize(2) 
data = reform(data,nx,ny,nz)

npx = sboxsize(0)
npy = sboxsize(1)
npz = sboxsize(2)

flatsize = npx*npy*npz
result=reform(data[0:(npx-1),0:(npy-1),0:(npz-1)],flatsize)
poslist = fltarr(3,nnxy*nnxy*nnz)
poslist(0,0) = lboxpos(0) + (npx-1.)/2.
poslist(1,0) = lboxpos(1) + (npy-1.)/2.
poslist(2,0) = lboxpos(2) + (npz-1.)/2.
kk = 1
for k = 0 , nnz -1 do begin
  for j = 0,nnxy-1 do begin
    for i = 0,nnxy-1 do begin
      if j eq 0 and i eq 0 and k eq 0 then continue
      temp=reform(data[ i*stepxy:(i*stepxy+npx-1),j*stepxy:(j*stepxy+npy-1),k*stepz:(k*stepz+npz-1)],flatsize)
      result = [[result],[temp]]
      poslist(0,kk) = lboxpos(0) + (npx-1.)/2. + i*stepxy
      poslist(1,kk) = lboxpos(1) + (npy-1.)/2. + j*stepxy
      poslist(2,kk) = lboxpos(2) + (npz-1.)/2. + k*stepz
      kk=kk+1
    endfor
  endfor
endfor
return,result
end 

;--------------------------------------------------------------------
; data here nned to be an 2d array,[*,i] is the pixel lightness, and i is the large box indice
function padbox2little_3dall,data=data,lboxsize,lboxpos,sboxsize,stepxy=stepxy,stepz=stepz,nnxy=nnxy,nnz=nnz,poslist = poslist

n = n_elements(data(0,*))
result = padbox2little_3d1(data=data(*,0),lboxsize,lboxpos(*,0),sboxsize,stepxy=stepxy,stepz=stepz,nnxy=nnxy,nnz=nnz,poslist = poslist)

for ii = 1 , n-1 do begin
   temp = padbox2little_3d1(data=data(*,ii),lboxsize,lboxpos(*,ii),sboxsize,stepxy=stepxy,stepz=stepz,nnxy=nnxy,nnz=nnz,poslist = temppos)
   result = [[result],[temp]]
   poslist = [[poslist],[temppos]]
endfor

return,result

end


























