       program linear_simple2
       implicit double precision (a-h,o-z)

       real*8,allocatable,dimension(:,:) :: feat_case0,feat_case
       real*8,allocatable,dimension(:,:) :: dd_all
       real*8,allocatable,dimension(:) :: Ei_case,Ei_ref
       real*8,allocatable,dimension(:,:) :: Gfeat_case
       integer,allocatable,dimension(:) :: ind_ref
       real*8,allocatable,dimension(:) :: dist_ref
       real*8,allocatable,dimension(:,:) :: S


       real*8,allocatable,dimension(:) :: work,BB
       real*8,allocatable,dimension(:) :: E_fit,W,Wc,Wc0
       integer,allocatable,dimension(:) :: ipiv
       integer lwork
       integer iatom_type(10)

       real*8,allocatable,dimension(:,:) :: featCR,featC
       

        
        write(6,*) "input itype,fact_4th"
        read(5,*) itype,fact_4th


       open(10,file="feat_new_stored."//char(48+itype),
     &      form="unformatted")
       rewind(10)
       read(10) num_case,nfeat0
       write(6,*) "num_case,nfeat", num_case,nfeat0
       allocate(Ei_case(num_case))
       allocate(feat_case0(nfeat0,num_case))
       do ii=1,num_case
       read(10) jj,Ei_case(ii),feat_case0(:,ii)
       enddo
       close(10)


       allocate(Wc(num_case))
       allocate(Wc0(num_case))
       allocate(feat_case(nfeat0,num_case))
       Wc=1.d0

        nfeat=nfeat0
        num_ref=nfeat

       allocate(S(num_ref,num_ref))
       allocate(BB(num_ref))
       allocate(ipiv(num_ref))
       allocate(E_fit(num_case))

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc     
!    In the following, we will do a linear fitting E= \sum_i W(i) feat2(i)
!    We will do several times, so have a average for W(i)^2. 
!    The everage W(i) will be used as a metrix to measure the distrance
!    Between two points. 

        do 100 kkk=1,200


        do ii=1,num_case
        do j=1,nfeat0
        feat_case(j,ii)=Wc(ii)*feat_case0(j,ii)
        enddo
        enddo



       S=0.d0
c       call dgemm('N','T',nfeat2,nfeat2,num_case(itype),1.d0,feat2_case,
c     & nfeat2,feat2_case,nfeat2,0.d0,S,nfeat2)
       call dgemm('N','T',num_ref,num_ref,num_case,1.d0,
     & feat_case,
     & num_ref,feat_case0,num_ref,0.d0,S,num_ref)

       do j=1,num_ref
       S(j,j)=S(j,j)+0.001
       enddo

       do j=1,num_ref
       sum=0.d0
       do i=1,num_case
       sum=sum+Ei_case(i)*feat_case(j,i)
       enddo
       BB(j)=sum
       enddo


!cccccccccccc

       call dgesv(num_ref,1,S,num_ref,ipiv,BB,num_ref,info)  

!cccccccccccccccccccccccccccccccccccccccccccccccccc
       
       diff1=0.d0
       diff2=0.d0
       diff4=0.d0
       do i=1,num_case
       sum=0.d0
       do j=1,num_ref
       sum=sum+BB(j)*feat_case0(j,i)
       enddo
       E_fit(i)=sum
       diff1=diff1+abs(E_fit(i)-Ei_case(i))
       diff2=diff2+(E_fit(i)-Ei_case(i))**2
       diff4=diff4+(E_fit(i)-Ei_case(i))**4
       Wc(i)=(E_fit(i)-Ei_case(i))**2/0.1**2
       enddo
       Wc=Wc+diff2/num_case/0.1**2*fact_4th
       diff1=(diff1/num_case)
       diff2=dsqrt(diff2/num_case)
       diff4=dsqrt(dsqrt(diff4/num_case))
       write(6,"('diff1,2,4=',3(E10.3,1x))") diff1,diff2,diff4

       if(kkk.gt.2) then
 
       sum=0.d0
       do i=1,num_case
       sum=sum+abs(Wc(i)-Wc0(i))
       enddo
       write(6,*) "diff in Wc=",sum
       Wc=Wc0*0.9+Wc*0.1
       endif

       Wc0=Wc

100   continue

       
       open(10,file="E_fit.linear."//char(48+itype))
       rewind(10) 
       do i=1,num_case
       write(10,"(2(E14.7,1x))") Ei_case(i),E_fit(i)
       enddo
       close(10)

       stop
       end
