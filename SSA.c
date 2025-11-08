#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define N 2
#define num_salp 10
#define LB -10.0 //cận dưới
#define UB 10.0 //cận trên
#define MAX_ITER 100


double position[num_salp][N];
double fitness[num_salp];
double bestposition[N];
double bestfitness;

double objective_function(double *x){
    double sum = 0.0;
    for(int i=0;i<N;i++){
        sum+=x[i]*x[i];
    }
    return sum;
}
double rand01(){
    return (double)rand()/RAND_MAX; //giá trị ngẫu nhiên [0,1]
}
int main(){
    //khởi tạo bộ sinh ngẫu nhiên
    srand(time(NULL));

    //khởi tạo ngẫu nhiên vị trí cho từng salp
    for(int i=0;i<num_salp;i++){
        for(int j=0;j<N;j++){
            position[i][j]=LB + rand01()*(UB-LB);
        }
    }

    //tính fitness của từng salp
    for(int i=0;i<num_salp;i++){
        fitness[i] = objective_function(position[i]);
    }

    //tìm salp tốt nhất ban đầu
    bestfitness = __DBL_MAX__;
    for(int i=0;i<num_salp;i++){
        if(fitness[i]<bestfitness){
            bestfitness = fitness[i];
            for(int j=0;j<N;j++){
                bestposition[j] = position[i][j];
            }
        }
    }

    //in ra trạng thái ban đầu
    printf("Danh sach salp ban dau:\n");
    for(int i=0;i<num_salp;i++){
        printf("Salp %2d: ",i+1);
        for(int j=0;j<N;j++){
            printf("%8.3f ",position[i][j]);
        }
        printf(" | f = %8.3f\n",fitness[i]);
    }

    printf("\nBest fitness ban dau: %.6f\n",bestfitness);
    printf("Best position : (");
    for(int j=0; j<N;j++){
        printf("%8.3f",bestposition[j]);
        if(j<N-1) printf(", ");
    }
    printf(")\n");

    for(int iter=0;iter<MAX_ITER;iter++){
        double c1 = 2*exp(-pow((4.0*iter / MAX_ITER),2)); //hệ số giảm dần

        for(int i=0;i<num_salp;i++){
            for(int j=0;j<N;j++){
                double c2 = rand01();
                double c3 = rand01();

                if(i==0)
                    {
                    //leader cập nhật theo best position
                    if(c3>=0.5) 
                        position[i][j] = bestposition[j] + c1*((UB - LB)*c2 + LB);
                    else 
                        position[i][j] = bestposition[j] - c1*((UB - LB)*c2 + LB);
                    }
                else 
                    {
                    //Follower : trung bình so với salp trước
                    position[i][j] = (position[i][j] + position[i-1][j])/2.0;
                    }
                if(position[i][j] < LB) position[i][j] = LB;
                if(position[i][j] > UB) position[i][j] = UB;
            }
        }

        //Đánh giá lại fitness
        for(int i=0;i<num_salp;i++){
            fitness[i] = objective_function(position[i]);
            if(fitness[i]<bestfitness){
                bestfitness = fitness[i];
                for(int j=0;j<N;j++){
                    bestposition[j] = position[i][j];
                }
            }
        }
    }
    //Ket thuc va in ket qua
    printf("\nBest fitness cuoi cung: %.6f\n",bestfitness);
    printf("Best position = (");
    for(int j=0;j<N;j++){
        printf("%8.3f",bestposition[j]);
        if(j<N-1) printf(", ");
    }
    printf(")\n");
    return 0;

}
