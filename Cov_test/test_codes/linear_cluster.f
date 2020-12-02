       program linear_cluster
       implicit double precision (a-h,o-z)

       real*8,allocatable,dimension(:,:) :: feat_case0,feat_case
       real*8,allocatable,dimension(:,:) :: dd_all
       real*8,allocatable,dimension(:) :: Ei_case,Ei_ref
       real*8,allocatable,dimension(:,:) :: Gfeat_case
       integer,allocatable,dimension(:) :: ind_ref
       real*8,allocatable,dimension(:) :: dist_ref
       real*8,allocatable,dimension(:,:) :: S


       real*8,allocatable,dimension(:) :: work,BB
       real*8,allocatable,dimension(:) :: E_fit,W
       integer,allocatable,dimension(:) :: ipiv
       integer lwork
       integer iatom_type(10)

       real*8,allocatable,dimension(:,:) :: featCR,featC
       

        
        write(6,*) "input itype"
        read(5,*) itype


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

       open(11,file="classf_cluster."//char(48+itype)) 
       rewind(11)
       read(11,*) numCR,ndim
       allocate(featCR(ndim,numCR))
       do ii=1,numCR
       read(11,*) (featCR(jj,ii),jj=1,ndim)
       enddo
       close(11)

       open(12,file="classf_all."//char(48+itype))
       rewind(12)
       read(12,*) num_case2,ndim2
       if(num_case.ne.num_case2) then
       write(6,*) "Two num_case not the same, stop", num_case,num_case2
       stop
       endif
       if(ndim.ne.ndim2) then
       write(6,*) "two ndim not the same, stop:,ndim,ndim2"
       stop
       endif
       allocate(featC(ndim,num_case))
       do ii=1,num_case
       read(12,*) (featC(jj,ii),jj=1,ndim)
       enddo
       close(12)

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc     
ccccc expand the feature from nfeat0 to nfeat=nfeat0*numCR
       allocate(dd_all(numCR,num_case))
       sum0=0.d0
       do ii=1,num_case
       do ii2=1,numCR
       sum=0.d0
       do jj=1,ndim
       sum=sum+(featC(jj,ii)-featCR(jj,ii2))**2
       enddo
       dd_all(ii2,ii)=sum
       sum0=sum0+sum
       enddo
       enddo

       dd_ave=sum0/(num_case*numCR)


       allocate(w(numCR))
       nfeat=nfeat0*numCR
       allocate(feat_case(nfeat,num_case))

        num_ref=nfeat

       allocate(S(num_ref,num_ref))
       allocate(BB(num_ref))
       allocate(ipiv(num_ref))
       allocate(E_fit(num_case))

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       fact=4
       mm=2
        write(6,*) "dd_ave=",dd_ave

        write(6,*) 
     & "input imth(1:exp;2:poly),(dd^mm):mm,(scan fac):fact1,fact2"
        read(5,*) imth,mm,fact1,fact2

       fact3=(fact2/fact1)**(1/20.d0)
       do 1000 kkk=1,20

       fact=fact1*fact3**kkk
       write(6,*) "kkk,mm,fact",kkk,mm,fact

       do ii=1,num_case

       sum=0.d0
       do ii2=1,numCR

       if(imth.eq.1) then
       xx=abs(dd_all(ii2,ii)/(dd_ave/fact))**mm
       if(xx.gt.100.d0) xx=100.d0
       w(ii2)=exp(-xx)
       else
       w(ii2)=1.d0/(dd_all(ii2,ii)**mm+(dd_ave/fact)**mm)
       endif

       sum=sum+w(ii2)
       enddo
       w=w/sum
      
       ifeat=0
       do jj=1,nfeat0
       do ii2=1,numCR
       ifeat=ifeat+1
       feat_case(ifeat,ii)=feat_case0(jj,ii)*w(ii2)
       enddo
       enddo
      
       enddo

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc     
!    In the following, we will do a linear fitting E= \sum_i W(i) feat2(i)
!    We will do several times, so have a average for W(i)^2. 
!    The everage W(i) will be used as a metrix to measure the distrance
!    Between two points. 


       S=0.d0
c       call dgemm('N','T',nfeat2,nfeat2,num_case(itype),1.d0,feat2_case,
c     & nfeat2,feat2_case,nfeat2,0.d0,S,nfeat2)
       call dgemm('N','T',num_ref,num_ref,num_case,1.d0,
     & feat_case,
     & num_ref,feat_case,num_ref,0.d0,S,num_ref)

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
       sum=sum+BB(j)*feat_case(j,i)
       enddo
       E_fit(i)=sum
       diff1=diff1+abs(E_fit(i)-Ei_case(i))
       diff2=diff2+(E_fit(i)-Ei_case(i))**2
       diff4=diff4+(E_fit(i)-Ei_case(i))**4
       enddo
       diff1=(diff1/num_case)
       diff2=dsqrt(diff2/num_case)
       diff4=dsqrt(dsqrt(diff4/num_case))
       write(6,"('diff1,2,4=',3(E10.3,1x))") diff1,diff2,diff4
1000   continue

       
       open(10,file="E_fit.linear."//char(48+itype))
       rewind(10) 
       do i=1,num_case
       write(10,"(2(E14.7,1x))") Ei_case(i),E_fit(i)
       enddo
       close(10)

       stop
       end
