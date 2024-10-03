#include<stdio.h>
#include<stdlib.h>
#include<conio.h>
#include<math.h>
#include<omp.h>

// Função sigmoid simplificada para usar no código
double sigmoid_simplified(double sum) {
	return 1.0/(1.0+(exp(-1*(sum))));
}

int main()
{
	double v;
	int i=0;
	int j,l;
	int n,m;
	int cs;
	int nSamples=1000;
	
	int nInputs=7;
	int nTargets=7;
	int nCases = 13612; //13612
	int nNeurons = 3;
	
	double er;
	double I0[7],O0[7],I1[7],O1[7],I2[7],O2[7];
	double  w1[100][100], w2[100][100];
	double nw1[100][100],nw2[100][100];
	double vw1[100][100],vw2[100][100];
	double d2[100],d1[100];
	
	// Create where inputs and targets are storage
	double E1[nCases];
	double E2[nCases];
	double E3[nCases];
	double E4[nCases];
	double E5[nCases];
	double E6[nCases];
	
	double t1[nCases];
	double t2[nCases];
	double t3[nCases];
	double t4[nCases];
	double t5[nCases];
	double t6[nCases];
	double t7[nCases];
	
	// Variaveis para acuracia e loss
	double cont=0;
	double loss=0;

//------------------LER-----------------------------
   	FILE *in;

	// Verifica se o arquivo existe
   	if ((in =fopen("./DryBeanDataset/dry_bean_data_input_random.txt", "rt"))== NULL)
//    if ((in =fopen("data_iris_training_2digits.txt", "rt"))== NULL)
	{
		printf("Cannot open input file.\n");
		return 1;
	}

	// Se existir vai salvar as entradas e suas respectivas saidas nos targets
   	while(!feof(in)){
		fscanf(in,"%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf",&E1[i],&E2[i],&E3[i],&E4[i],&E5[i],&t1[i],&t2[i],&t3[i],&t4[i],&t5[i],&t6[i],&t7[i]);
	  	i++;
	}

	// Mostra na tela o que acabou de escrever na memória
	for(j=0;j<(i-1);j++){
		printf("%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n",j,E1[j],E2[j],E3[j],E4[j],E5[j],t1[j],t2[j],t3[j],t4[j],t5[j],t6[j],t7[j]);
	}
	
	// Fecha o arquivo
   	fclose(in);
   	//free(in);
	getch();
	//-----------------------------------------------------
	
	// Inicia os pesos de w1 e w2 de modo aleatório
	for(j=1;j<=7;j++){
		for(i=1;i<=6;i++){
			w1[i][j]=((rand()%100)/10.0)-5.0;
			vw1[i][j]=0.0;
		}
	}
	
	for(l=1;l<=7;l++){
		for(j=1;j<=5;j++){
			w2[j][l]=((rand()%100)/10.0)-5.0;
			vw2[j][l]=0.0;
		}
	}
	
	// Inicia o treinamento da rede
	int nTrain=1000;
	// Paraleliza o loop com OpenMP
    #pragma omp parallel for private[m]
	for(m=0;m<=nTrain;m++){
		er=0.0;
		for(n=0;n<=10000;n++){
		
			//-------------------propagation--------------
			// Gera um número randômico entre 0 e nCases
			cs=rand()%nCases;
			
			I0[1]=E1[cs];
			I0[2]=E2[cs];
			I0[3]=E3[cs];
			I0[4]=E4[cs];
			I0[5]=E5[cs];
			I0[6]=1.0;
			
			// Passa a entrada para O para I
			O0[1]=I0[1];
			O0[2]=I0[2];
			O0[3]=I0[3];
			O0[4]=I0[4];
			O0[5]=I0[5];
			O0[6]=I0[6];
			
			// Aplica os pesos nas entradas e a sigmoid
			for(j=1;j<=6;j++){
			  	I1[j]=0.0;
				for(i=1;i<=6;i++){
				  I1[j]+=O0[i]*w1[i][j];
				}
				
				O1[j]=sigmoid_simplified(I1[j]);
			}
			
			I1[5]=1.0;
			O1[5]=I1[5];
			
			for(l=1;l<=7;l++){
				I2[l]=0.0;
				for(j=1;j<=7;j++){
					I2[l]+=O1[j]*w2[j][l];
				}
				
				O2[l]=sigmoid_simplified(I2[l]);
			}
			
			
			//----------backpropagation--------------------------------
			
			d2[1]=(t1[cs]-O2[1])*O2[1]*(1.0-O2[1]);
			d2[2]=(t2[cs]-O2[2])*O2[2]*(1.0-O2[2]);
			d2[3]=(t3[cs]-O2[3])*O2[3]*(1.0-O2[3]);
			d2[4]=(t4[cs]-O2[4])*O2[4]*(1.0-O2[4]);
			d2[5]=(t5[cs]-O2[5])*O2[5]*(1.0-O2[5]);
			d2[6]=(t6[cs]-O2[6])*O2[6]*(1.0-O2[6]);
			d2[7]=(t7[cs]-O2[7])*O2[7]*(1.0-O2[7]);
			
			d1[1]=O1[1]*(1.0-O1[1])*(d2[1]*w2[1][1]+d2[2]*w2[1][2]+d2[3]*w2[1][3]);
			d1[2]=O1[2]*(1.0-O1[2])*(d2[1]*w2[2][1]+d2[2]*w2[2][2]+d2[3]*w2[2][3]);
			d1[3]=O1[3]*(1.0-O1[3])*(d2[1]*w2[3][1]+d2[2]*w2[3][2]+d2[3]*w2[3][3]);
			d1[4]=O1[4]*(1.0-O1[4])*(d2[1]*w2[4][1]+d2[2]*w2[4][2]+d2[3]*w2[4][3]);
			d1[5]=O1[5]*(1.0-O1[5])*(d2[1]*w2[5][1]+d2[2]*w2[5][2]+d2[3]*w2[5][3]);
			d1[6]=O1[6]*(1.0-O1[6])*(d2[1]*w2[6][1]+d2[2]*w2[6][2]+d2[3]*w2[6][3]);
			d1[7]=O1[7]*(1.0-O1[7])*(d2[1]*w2[7][1]+d2[2]*w2[7][2]+d2[3]*w2[7][3]);
			
			// Aplica uma correção baseada no gradiente descendente
			for(l=1;l<=7;l++){
				for(j=1;j<=7;j++){
					nw2[j][l]= w2[j][l]+0.5*d2[l]*O1[j]+0.5*vw2[j][l];
					vw2[j][l]=nw2[j][l]-w2[j][l];
					w2[j][l]=nw2[j][l];
				}
			}
			
			//-------------------------------------------------
			for(j=1;j<=6;j++){
				for(i=1;i<=6;i++){
					nw1[i][j]= w1[i][j]+0.5*d1[j]*O0[i]+0.5*vw1[i][j];
					vw1[i][j]=nw1[i][j]-w1[i][j];
					w1[i][j]=nw1[i][j];
				}
			}
			//-----------------------------------------------
			// Faz o calculo do erro
			er+= (t1[cs]-O2[1])*(t1[cs]-O2[1])+
				 (t2[cs]-O2[2])*(t2[cs]-O2[2])+
				 (t3[cs]-O2[3])*(t3[cs]-O2[3])+
				 (t4[cs]-O2[4])*(t4[cs]-O2[4])+
				 (t5[cs]-O2[5])*(t5[cs]-O2[5])+
				 (t6[cs]-O2[6])*(t6[cs]-O2[6])+
				 (t7[cs]-O2[7])*(t7[cs]-O2[7]);
		}//n
		er=er/10000.0;
		printf("%d %lf \n",m,er);
	}//m
	
	// Mostra os resultaddos dos pesos encontrados
	printf("\nPesos encontrados: \n");
	for(j=1;j<=6;j++){
		for(i=1;i<=6;i++){
		 	printf("%f ",w1[i][j]);
		}
		printf("\n");
	}
	
	for(l=1;l<=7;l++){
		for(j=1;j<=7;j++){
	 		printf("%f ",w2[j][l]);		
		}
		printf("\n");
	}
	printf("\n=================================\n");
	//-----------------------------------------

	// Verifica se o arquivo existe de teste
   	if ((in =fopen("./DryBeanDataset/dry_bean_data_input_random.txt", "rt"))== NULL)
   	{
	  printf("Cannot open input file.\n");
	  return 1;
   	}
	
	i=0;
	while(!feof(in)){
		fscanf(in,"%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf",&E1[i],&E2[i],&E3[i],&E4[i],&E5[i],&t1[i],&t2[i],&t3[i],&t4[i],&t5[i],&t6[i],&t7[i]);
	  	i++;
	}

	// Mostra na tela o que acabou de escrever na memória
	for(j=0;j<(i-1);j++){
		printf("%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",j,E1[j],E2[j],E3[j],E4[j],E5[j],t1[j],t2[j],t3[j],t4[j],t5[j],t6[j],t7[j]);
	}
	
	// Fecha o arquivo
   	fclose(in);
	getch();
	
	// Verificacao dos resultados da rede para todas as amostras
	for(m=0;m<=nSamples;m++){
	
		I0[1]=E1[m];
		I0[2]=E2[m];
		I0[3]=E3[m];
		I0[4]=E4[m];
		I0[5]=E5[m];
		I0[6]=1.0;
		
		// Passa a entrada para O para I
		O0[1]=I0[1];
		O0[2]=I0[2];
		O0[3]=I0[3];
		O0[4]=I0[4];
		O0[5]=I0[5];
		O0[6]=I0[6];
		
		// Calculo das ativacoes da camada intermediaria
		for(j=1;j<=6;j++){
		  	I1[j]=0.0;
			for(i=1;i<=6;i++){
			  I1[j]+=O0[i]*w1[i][j];
			}
			O1[j]=sigmoid_simplified(I1[j]);
		}
		I1[5]=1.0;
		O1[5]=I1[5];
		
		// Calculo das ativacoes da camada de saida
		
		for(l=1;l<=7;l++){
			I2[l]=0.0;
			for(j=1;j<=7;j++){
				I2[l]+=O1[j]*w2[j][l];
			}
			O2[l]=sigmoid_simplified(I2[l]);
		}

		// If para contar a acuracia e o loss da amostra 1 
		if(O2[1]>O2[2] && O2[1]>O2[3] && O2[1]>O2[4] && O2[1]>O2[5] && O2[1]>O2[6] && O2[1]>O2[7]) {
			cont +=1;
			loss += (-log(O2[1]));
		}
		// Saidas
		//saidas
	}
	
	printf("\n-------------------- Acuracia e Loss Amostra 0-49 --------------------\n");
	printf("Acuracia : %f", cont/(double)nSamples*100.0);
	printf("\nloss : %f", loss);
	
	printf("\n\n-------------------------------------------------------------------------------------\n");
}
