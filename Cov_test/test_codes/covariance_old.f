       program covariance
       implicit double precision (a-h,o-z)

       real*8,allocatable,dimension(:,:) :: feat_case0,feat_case
       real*8,allocatable,dimension(:,:) :: dd_all
       real*8,allocatable,dimension(:) :: Ei_case,Ei_ref
       real*8,allocatable,dimension(:,:) :: Gfeat_case
       integer,allocatable,dimension(:) :: ind_ref
       real*8,allocatable,dimension(:) :: dist_ref
       real*8,allocatable,dimension(:,:) :: S,dd,S2
       real*8,allocatable,dimension(:) :: weight
       real*8,allocatable,dimension(:,:) :: feat_rp


       real*8,allocatable,dimension(:) :: work,BB
       real*8,allocatable,dimension(:) :: E_fit,W,cov
       integer,allocatable,dimension(:) :: ipiv
       integer lwork
       integer iatom_type(10)

       real*8,allocatable,dimension(:,:) :: featCR,featC
       real*8,allocatable,dimension(:) :: SS1,SS2
      

        
        write(6,*) "input itype"
        read(5,*) itype
        write(6,*) "input param: num_rpm,ave_dive (for ave/ave_div)"
        read(5,*) num_rp,ave_dive


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

       allocate(weight(nfeat0))
       open(11,file="weight_feat."//char(48+itype)) 
       rewind(11)
       do j=1,nfeat0
       read(11,*) j1,weight(j)
       enddo
       close(11)


       allocate(feat_rp(nfeat0,num_rp))

       do ii=1,num_case
       do j=1,nfeat0
       feat_case0(j,ii)=feat_case0(j,ii)*weight(j)
       enddo
       enddo
       
       nn=num_case/num_rp-1
       do ii=1,num_rp
       icase=(ii-1)*nn+1
       do j=1,nfeat0
       feat_rp(j,ii)=feat_case0(j,icase)
       enddo
       enddo

       allocate(S(num_rp,num_rp))
       allocate(dd(num_rp,num_rp))
   
       call dgemm('T','N',num_rp,num_rp,nfeat0,1.d0,
     & feat_rp,
     & nfeat0,feat_rp,nfeat0,0.d0,S,num_rp)

       ave_dd=0.d0
       do i1=1,num_rp
       do i2=1,num_rp
       dd(i1,i2)=abs(S(i1,i1)+S(i2,i2)-2*S(i1,i2))   ! the distance
       ave_dd=ave_dd+dd(i1,i2)
       enddo
       enddo
       ave_dd=ave_dd/(num_rp**2)
       write(6,*) "ave_dd=",ave_dd

       do i1=1,num_rp
       do i2=1,num_rp
!       dd(i1,i2)=1.d0/(dd(i1,i2)+ave_dd/20)
       dd(i1,i2)=1.d0/(dd(i1,i2)+ave_dd/ave_dive)
       enddo
       enddo


       do i1=1,num_rp
       dd(i1,i1)=dd(i1,i1)+(0.01/ave_dd)**2
       enddo

       lwork=num_rp
       allocate(ipiv(num_rp))
       allocate(work(lwork))

ccccccccccccccc inversion
       call dgetrf(num_rp,num_rp,dd,num_rp,ipiv,info) 
       call dgetri(num_rp,dd,num_rp,ipiv,work,lwork,info)

       deallocate(ipiv)
       deallocate(work)
       deallocate(S)
       allocate(S(num_rp,num_case))
       allocate(S2(num_rp,num_case))
       call dgemm('T','N',num_rp,num_case,nfeat0,1.d0,
     & feat_rp,
     & nfeat0,feat_case0,nfeat0,0.d0,S,num_rp)


       allocate(SS1(num_rp))
       allocate(SS2(num_case))
       do i1=1,num_rp
       sum=0.d0
       do j=1,nfeat0
       sum=sum+feat_rp(j,i1)**2
       enddo
       SS1(i1)=sum
       enddo

       do i1=1,num_case
       sum=0.d0
       do j=1,nfeat0
       sum=sum+feat_case0(j,i1)**2
       enddo
       SS2(i1)=sum
       enddo


       do i1=1,num_rp
       do i2=1,num_case
       S(i1,i2)=abs(SS1(i1)+SS2(i2)-2*S(i1,i2))
!       S(i1,i2)=1.d0/(S(i1,i2)+ave_dd/20)
       S(i1,i2)=1.d0/(S(i1,i2)+ave_dd/ave_dive)
       enddo
       enddo
       
       call dgemm('N','N',num_rp,num_case,num_rp,1.d0,
     & dd,num_rp,S,num_rp,0.d0,S2,num_rp)

       allocate(cov(num_case))

       do i2=1,num_case
       sum=0.d0
       do i1=1,num_rp
       sum=sum+S(i1,i2)*S2(i1,i2)
       enddo
       cov(i2)=sum
       enddo

       
       open(10,file="test")
       rewind(10) 
       do i=1,num_case
       write(10,"(2(E14.7,1x))") Ei_case(i),cov(i)
       enddo
       close(10)

       stop
       end
