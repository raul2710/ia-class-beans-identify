#include<stdio.h>
#include<stdlib.h>
#include<conio.h>
#include<math.h>

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
	double er;
	double I0[15],O0[15],I1[15],O1[15],I2[13],O2[13];
	double  w1[15][14] ,w2[15][13];
	double nw1[15][14],nw2[15][13];
	double vw1[15][14],vw2[15][13];
	double d2[13],d1[14];
	
	// Cria onde serão armazenadas as entradas e os targets
	double E1[13612];
	double E2[13612];
	double E3[13612];
	double E4[13612];
	double E5[13612];
	double E6[13612];
	double E7[13612];
	
	double t1[13612];
	double t2[13612];
	double t3[13612];
	double t4[13612];
	double t5[13612];
	double t6[13612];
	double t7[13612];
	
	// Variaveis para acuracia e loss
	double cont=0;
	double loss=0;

//------------------LER-----------------------------
   	FILE *in;

	// Verifica se o arquivo existe
   	if ((in =fopen("data_iris_training_4digits.txt", "rt"))== NULL)
//    if ((in =fopen("data_iris_training_2digits.txt", "rt"))== NULL)
	{
		printf("Cannot open input file.\n");
		return 1;
	}

	// Se existir vai salvar as entradas e suas respectivas saidas nos targets
   while(!feof(in)){
	  fscanf(in,"%lf%lf%lf%lf%lf%lf%lf",&E1[i],&E2[i],&E3[i],&E4[i],&t1[i],&t2[i],&t3[i]);
	  i++;
	}

	// Mostra na tela o que acabou de escrever na memória
	for(j=0;j<(i-1);j++){
		printf("%d %f %f %f %f %f %f %f\n",j,E1[j],E2[j],E3[j],E4[j],t1[j],t2[j],t3[j]);
	}
	
	// Fecha o arquivo
   	fclose(in);
   	//free(in);
	getch();
	//-----------------------------------------------------
	
	// Inicia os pesos de w1 e w2 de modo aleatório
	for(j=1;j<=3;j++){
		for(i=1;i<=5;i++){
			w1[i][j]=((rand()%100)/10.0)-5.0;
			vw1[i][j]=0.0;
		}
	}
	
	for(l=1;l<=3;l++){
		for(j=1;j<=4;j++){
			w2[j][l]=((rand()%100)/10.0)-5.0;
			vw2[j][l]=0.0;
		}
	}
	
	// Inicia o treinamento da rede
	for(m=0;m<=1000;m++){
		er=0.0;
		for(n=0;n<=10000;n++){
		
			//-------------------propagation--------------
			// Gera um número randômico entre 0 e 105
			cs=rand()%105;
			
			I0[1]=E1[cs];
			I0[2]=E2[cs];
			I0[3]=E3[cs];
			I0[4]=E4[cs];
			I0[5]=1.0;
			
			// Passa a entrada para O para I
			// O0 -> I0
			// O1 -> I1
			// O2 -> I2
			// O3 -> I3
			// O4 -> I4
			// O5 -> I5
			O0[1]=I0[1];
			O0[2]=I0[2];
			O0[3]=I0[3];
			O0[4]=I0[4];
			O0[5]=I0[5];
			
			// Aplica os pesos nas entradas e a sigmoid
			for(j=1;j<=3;j++){
			  	I1[j]=0.0;
				for(i=1;i<=5;i++){
				  I1[j]+=O0[i]*w1[i][j];
				}
				
				O1[j]=sigmoid_simplified(I1[j]);
			}
			
			I1[4]=1.0;
			O1[4]=I1[4];
			
			for(l=1;l<=3;l++){
				I2[l]=0.0;
				for(j=1;j<=4;j++){
					I2[l]+=O1[j]*w2[j][l];
				}
				O2[l]=sigmoid_simplified(I2[l]);
			}
			
			
			//----------backpropagation--------------------------------
			
			// Inicia capturando o erro d2 e logo 
			// em seguida pega o erro d1 sempre de trás para frente
			d2[1]=(t1[cs]-O2[1])*O2[1]*(1.0-O2[1]);
			d2[2]=(t2[cs]-O2[2])*O2[2]*(1.0-O2[2]);
			d2[3]=(t3[cs]-O2[3])*O2[3]*(1.0-O2[3]);
			
			d1[1]=O1[1]*(1.0-O1[1])*(d2[1]*w2[1][1]+d2[2]*w2[1][2]+d2[3]*w2[1][3]);
			d1[2]=O1[2]*(1.0-O1[2])*(d2[1]*w2[2][1]+d2[2]*w2[2][2]+d2[3]*w2[2][3]);
			d1[3]=O1[3]*(1.0-O1[3])*(d2[1]*w2[3][1]+d2[2]*w2[3][2]+d2[3]*w2[3][3]);
			
			// Aplica uma correção baseada no gradiente descendente
			for(l=1;l<=3;l++){
				for(j=1;j<=4;j++){
					nw2[j][l]= w2[j][l]+0.5*d2[l]*O1[j]+0.5*vw2[j][l];
					vw2[j][l]=nw2[j][l]-w2[j][l];
					w2[j][l]=nw2[j][l];
				}
			}
			
			//-------------------------------------------------
			for(j=1;j<=3;j++){
				for(i=1;i<=5;i++){
					nw1[i][j]= w1[i][j]+0.5*d1[j]*O0[i]+0.5*vw1[i][j];
					vw1[i][j]=nw1[i][j]-w1[i][j];
					w1[i][j]=nw1[i][j];
				}
			}
			//-----------------------------------------------
			// Faz o calculo do erro
			er+= (t1[cs]-O2[1])*(t1[cs]-O2[1])+(t2[cs]-O2[2])*(t2[cs]-O2[2])+(t3[cs]-O2[3])*(t3[cs]-O2[3]);
		}//n
		er=er/10000.0;
		printf("%d %f \n",m,er);
	}//m
	
	// Mostra os resultaddos dos pesos encontrados
	printf("\nPesos encontrados: \n");
	for(j=1;j<=3;j++){
		for(i=1;i<=5;i++){
		 	printf("%f ",w1[i][j]);
		}
		printf("\n");
	}
	
	for(l=1;l<=3;l++){
		for(j=1;j<=4;j++){
	 		printf("%f ",w2[j][l]);		
		}
		printf("\n");
	}
	printf("\n=================================\n");
	//-----------------------------------------

	// Verifica se o arquivo existe de teste
	//if ((in =fopen("data_test_4digits.txt", "rt"))== NULL)
   	if ((in =fopen("data_test_2digits.txt", "rt"))== NULL)
   	{
	  printf("Cannot open input file.\n");
	  return 1;
   	}

	// Se existir vai salvar as entradas e suas respectivas saidas nos targets
   	while(!feof(in)){
	  fscanf(in,"%lf%lf%lf%lf%lf%lf%lf",&E1[i],&E2[i],&E3[i],&E4[i],&t1[i],&t2[i],&t3[i]);
	  i++;
	}

	// Mostra na tela o que acabou de escrever na memória
	for(j=0;j<(i-1);j++){
		printf("%d %f %f %f %f %f %f %f\n",j,E1[j],E2[j],E3[j],E4[j],t1[j],t2[j],t3[j]);
	}
	
	// Fecha o arquivo
   	fclose(in);
   	//free(in);
	getch();
	
	// Verificacao dos resultados da rede para todas as amostras
	for(m=0;m<=44;m++){
	
		I0[1]=E1[m];
		I0[2]=E2[m];
		I0[3]=E3[m];
		I0[4]=E4[m];
		I0[5]=1.0;
		
		O0[1]=I0[1];
		O0[2]=I0[2];
		O0[3]=I0[3];
		O0[4]=I0[4];
		O0[5]=I0[5];
		
		// Calculo das ativacoes da camada intermediaria
		for(j=1;j<=3;j++){
		  	I1[j]=0.0;
			for(i=1;i<=5;i++){
			  I1[j]+=O0[i]*w1[i][j];
			}
			O1[j]=sigmoid_simplified(I1[j]);
		}
		I1[4]=1.0;
		O1[4]=I1[4];
		
		// Calculo das ativacoes da camada de saida
		
		for(l=1;l<=3;l++){
			I2[l]=0.0;
			for(j=1;j<=4;j++){
				I2[l]+=O1[j]*w2[j][l];
			}
			O2[l]=sigmoid_simplified(I2[l]);
		}

		// If para contar a acuracia e o loss da amostra 1 
		if(O2[1]>O2[2] && O2[1]>O2[3]){
			cont +=1;
			loss += (-log(O2[1]));
		}
		// Saidas
		//saidas
//		printf("%d                                     %f %f %f \n",m,t1[m],t2[m],t3[m]);
		printf("%d %f %f %f %f %f %f %f \n",m,E1[m],E2[m],E3[m],E4[m],O2[1],O2[2],O2[3]);
	}
	
	printf("\n-------------------- Acuracia e Loss Amostra 0-49 --------------------\n");
	printf("Acuracia : %f", cont/45.0*100.0);
	printf("\nloss : %f", loss);
	
	printf("\n\n-------------------------------------------------------------------------------------\n");
}
