  !// forquill v1.01 beta www.fcode.cn
module read_dfeatnn
    !implicit double precision (a-h, o-z)
    implicit none
  
  !!!!!!!!!!!!!          以下为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    integer(4) :: m_neigh                                  !模型所使用的最大近邻数(考虑这个数是否可以不用)
    integer(4) :: nfeat0m                                  !不同种原子的原始feature数目中最大者(目前似无意义)
 
    
    integer(4) :: natom                                    !image的原子个数  
     
    
    real*8,allocatable,dimension(:,:) :: force       !每个原子的受力
    real*8,allocatable,dimension(:,:,:,:) :: dfeat
    real*8,allocatable,dimension(:,:) :: feat    
    real*8, allocatable,dimension(:) :: energy        !每个原子的能量
    integer(4),allocatable,dimension(:,:) :: list_neigh
    integer(4),allocatable,dimension(:) :: iatom
  !!!!!!!!!!!!!          以上为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
    contains

    subroutine deallo()
        deallocate(energy)
        deallocate(force)
        deallocate(feat)
        ! deallocate(num_neigh)
        deallocate(dfeat)
        deallocate(list_neigh)
        deallocate(iatom)
    end subroutine deallo
   
    subroutine read_dfeat(dfeatDir,image_Num,pos,itype_atom,rad_atom,wp_atom)
        integer(4)  :: image,nimage,nfeat0_tmp,jj,num_tmp,ii,i_p,i,j
        character(*),intent(in) :: dfeatDir
        integer(4), intent(in) :: image_Num
        integer(8), intent(in) :: pos
        real(8),dimension(:),intent(in) :: rad_atom, wp_atom 
        integer(4),dimension(:), intent(in) :: itype_atom
        real(8) :: dE,dEdd,dFx,dFy,dFz, rad1,rad2,rad, dd,yy, dx1,dx2,dx3,dx,dy,dz, pi,w22
        integer(4)  :: iitype, itype, ntype
        integer(4),allocatable, dimension (:) :: iatom_type
        real*8 AL(3,3)
        integer nfeat1tm(100),nfeat1t(100),ntype_tmp
        
        ! real*8,allocatable,dimension(:,:,:,:) :: dfeat
        ! real*8, allocatable,dimension(:,:) :: feat        
        real*8, allocatable, dimension (:,:) :: dfeat_tmp
        integer(4),allocatable, dimension (:) :: iat_tmp,jneigh_tmp,ifeat_tmp
        integer(4),allocatable,dimension(:) :: num_neigh    
        real*8,allocatable,dimension(:,:) :: xatom
        
        character(len=80) dfeatDirname

        dfeatDirname=trim(adjustl(dfeatDir))
        ! write(*,*) dfeatDirname
        ! open (23,file='/home/buyu/MLFF/AlHcomb_new/H2_hT/dfeat.fbin',action="read",form="unformatted",access='stream')
        open(23,file=trim(dfeatDirname),action="read",form="unformatted",access='stream')
        rewind(23)
        read(23) nimage,natom,nfeat0_tmp,m_neigh
        read(23) ntype_tmp,(nfeat1t(ii),ii=1,ntype_tmp)
        allocate(iatom(natom))
        read(23) iatom
        ! write(*,*) nimage,natom,nfeat0_tmp,m_neigh
        ! open(99,file='log.txt')         
        nfeat0m=nfeat0_tmp
         allocate(energy(natom))
         allocate(force(3,natom))
         allocate(feat(nfeat0m,natom))
         allocate(num_neigh(natom))
         allocate(list_neigh(m_neigh,natom))
         allocate(dfeat(nfeat0m,natom,m_neigh,3))
         allocate(xatom(3,natom))

    ! do 3000 image=1,image_Num
            ! write(*,*) image

        
     
        ! if (image.eq.image_Num) then
            read(23,pos=pos) energy
            
            read(23) force
            ! force = force*(-1)
            ! write(*,*) force(1,10)
            read(23) feat
            ! Inquire( 12 , Pos = i_p )
            ! write(*,*) i_p
            ! write(*,*) trim(dfeatDirname)
            ! write(*,*) image
            ! write(*,*) feat(19,1)
            read(23) num_neigh
            read(23) list_neigh
     !TODO:
            ! read(23) dfeat
            read(23) num_tmp
            ! write(*,*) num_tmp
            allocate(dfeat_tmp(3,num_tmp))
            allocate(iat_tmp(num_tmp))
            allocate(jneigh_tmp(num_tmp))
            allocate(ifeat_tmp(num_tmp))
            read(23) iat_tmp
            ! write(*,*) iat_tmp(3)
            read(23) jneigh_tmp
            read(23) ifeat_tmp
            read(23) dfeat_tmp
            
            read(23) xatom    ! xatom(3,natom)
            read(23) AL       ! AL(3,3)
            ! allocate(dfeat(nfeat0_tmp,natom,m_neigh,3))
            dfeat(:,:,:,:)=0.0
            do jj=1,num_tmp
            dfeat(ifeat_tmp(jj),iat_tmp(jj),jneigh_tmp(jj),:)=dfeat_tmp(:,jj)
            enddo
            deallocate(dfeat_tmp)
            deallocate(iat_tmp)
            deallocate(jneigh_tmp)
            deallocate(ifeat_tmp)
            close(23)

            
            do jj=1,natom
                ii=num_neigh(jj)+1
                list_neigh(ii:,jj)=0

            enddo

            pi=4*datan(1.d0)
            ntype=size(itype_atom)
            allocate(iatom_type(natom))
            do i=1,natom
                iitype=0
                do itype=1,ntype
                if(itype_atom(itype).eq.iatom(i)) then
                iitype=itype
                endif
                enddo
                if(iitype.eq.0) then
                write(6,*) "this type not found", iatom(i)
                endif
                iatom_type(i)=iitype
            enddo
            do i=1,natom
                rad1=rad_atom(iatom_type(i))
                dE=0.d0
                dFx=0.d0
                dFy=0.d0
                dFz=0.d0
                do jj=1,num_neigh(i)
                j=list_neigh(jj,i)
                if(i.ne.j) then
                rad2=rad_atom(iatom_type(j))
                rad=rad1+rad2
                dx1=mod(xatom(1,j)-xatom(1,i)+100.d0,1.d0)
                if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1
                dx2=mod(xatom(2,j)-xatom(2,i)+100.d0,1.d0)
                if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1
                dx3=mod(xatom(3,j)-xatom(3,i)+100.d0,1.d0)
                if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1
                dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
                dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
                dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3
                dd=dsqrt(dx**2+dy**2+dz**2)
                if(dd.lt.2*rad) then
         !       write(6,"(2(i4,1x),3(f10.5,1x),2x,f13.6)") i,j,dx1,dx2,dx3,dd
                w22=dsqrt(wp_atom(iatom_type(i))*wp_atom(iatom_type(j)))
                yy=pi*dd/(4*rad)
         !       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
         !       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2
         !     &   -(pi/(2*rad))*cos(yy)*sin(yy))
         
         
                dE=dE+0.5*4*w22*(rad/dd)**12*cos(yy)**2
                dEdd=4*w22*(-12*(rad/dd)**12/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)
                dFx=dFx-dEdd*dx/dd       ! note, -sign, because dx=d(j)-x(i)
                dFy=dFy-dEdd*dy/dd
                dFz=dFz-dEdd*dz/dd
                endif
                endif
                enddo
         !       write(6,*) "dE,dFx",dE,dFx
                energy(i)=energy(i)-dE
                force(1,i)=force(1,i)-dFx   ! Note, assume force=dE/dx, no minus sign
                force(2,i)=force(2,i)-dFy
                force(3,i)=force(3,i)-dFz
            enddo
        deallocate(iatom_type)
        ! deallocate(feat)
        ! deallocate(energy)
        !   deallocate(force)
        ! deallocate(feat)
        deallocate(num_neigh)
        ! deallocate(iatom)
        deallocate(xatom)



    end subroutine read_dfeat
  

end module read_dfeatnn
  
  
