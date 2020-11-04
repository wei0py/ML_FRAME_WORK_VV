       subroutine kmean(feat_all,ncase,ind_cluster,feat_cluster,num_cluster)
       implicit double precision (a-h,o-z)

       real*8 feat_all(10,ncase),feat_cluster(10,200)
       real*8 feat_cent(10,200),dist_tmp(200)
       integer ind_cluster(10,200),ncount(200),ncount_cluster(200)

       do jj=1,num_cluster
       do i=1,8
       feat_cent(i,jj)=ind_cluster(i,jj)+0.5   ! initial
       enddo
       enddo

       do 1000 iter=1,10

       feat_cluster=0.d0
       ncount_cluster=0

       do 100 icase=1,ncase
       dist_min=1.D+30
       do jj=1,num_cluster
       sum=0.d0
       do i=1,8
       sum=sum+(feat_all(i,icase)-feat_cent(i,jj))**2
       enddo
       if(sum.lt.dist_min) then
       dist_min=sum
       jjmin=jj
       endif
       enddo
       feat_cluster(:,jjmin)=feat_cluster(:,jjmin)+feat_all(:,icase)
       ncount_cluster(jjmin)=ncount_cluster(jjmin)+1
100    continue

       diff=0.d0
       do jj=1,num_cluster
       do i=1,8
       feat_cluster(i,jj)=feat_cluster(i,jj)/ncount_cluster(jj)
       diff=diff+(feat_cluster(i,jj)-feat_cent(i,jj))**2
       feat_cent(i,jj)=feat_cluster(i,jj)
       enddo
       enddo
       write(6,*) "feature cluster diff",iter,diff
1000   continue

       do jj=1,num_cluster
       write(6,*) "ncount,cluster", ncount_cluster(jj)
       enddo

       return
       end subroutine kmean
      
       
