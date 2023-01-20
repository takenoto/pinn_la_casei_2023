# Discussion

# TODOs novos
- Ver influência da relação peso (weight) vs constante de adimensionalidade

## Conclusões
- O que discutir nos tópicos do trabalho:
    - Relação entre pesos (testar um a um), constantes de adimensionalidade e convergência
    - Relação entre fatores de adimensionalidade/escala (testar um a um) e convergência
        - O volume já deu pra ver que quanto maior scaler.V pior é a solução final, menos condizente
        com a realidade
        - O problema NÃO é o fato de não corrigir a concentração (que é m/V) porque se fosse pra corrigir o volume dessa eq, teria que corrigir também o fator de escala da massa, e ia ser justamente o inverso do volume, fim.

## TODO
    - PRIORIDADE: GRIDSEARCH FUNCIONAL
    - PRIORIDADE 2: Como salvar os arquivos?
    - TODO faça primeiro só 2 e veja como se sai o loop
- Várias vezes a única coisa errada é o volume, que sobe ou cai erroneamente e faz as concentrações parecerem constantes ou irem para locais errados, mas isso é tudo em função da variação sem sentido (não consegue convergir o volume). Testar aumentar o peso dele.
- Erro: modo com NonDim não faz sentido.
    - Fig2 chega n estacionario em 8h sem o non_dim.
        - Resolvido com modificações. Parece ser ainda aquele bug entre versões específicas do numpy e do tensorflow. Resolvi assim:
        `
        non_dim_rX = (
                   mult(mult( mult(div(scaler.t, scaler.X)
                    , mu_max)
                    , mult(X , scaler.X))
                    , div(mult(S,scaler.S),add(K_S, mult(S,scaler.S))))
                    * f_x_calc_func()
                    * h_p_calc_func()
                )
        `
    - Para NonDim com tudo = 1 deveria ser EXATAMENTE o mesmo resultado e não tá dando
    Teste mais, veja onde causa.
    - Testes já realizados:
        - Xo=0, tudo (X, P, S e V) permanece constante num reator batelada. OK!
        - zerar manualmente a derivada de dx OK
        - zerar uma a uma as derivadas, tudo fica reta. O problema realmente parece ser no rX.
- Implementar algum método numérico, como euler
    - Tem como usar as derivadas da minha própria função? Tipo mando os valores iniciais, pego as derivads, simplesmente multiplico por dt e somo cada uma no seu lugar.
    - Dá não pq eu retorno é a diferença entre elas e o previsto. Vou ter que reescrever mesmo...
- Estudar adimensionalização
    - Testar fazer a adimensionalização de forma simples, com um objeto que é passado e que sozinh se resolve (métodos to nondim e fromnondim para converter e desconverter, aí diz só se é X, S, V, etc)
- Descobrir como pegar o r² médio do PINN pra selecionar o melhor
- Criar arquivo com os valores experimentais para plotar tudo junto no fim e comparar
- Plotar os 3 e comparar
- Falar com amaro sobre bugs específicos do tensorflow... melhor pytorch??? Olha esse arrudeio pro erro do numpy, depende totalmente da versão: https://stackoverflow.com/questions/50678620/typeerror-input-b-of-matmul-op-has-type-float32-that-does-not-match-type-in

## Não entendi bem
- A loss é a soma dos ² da diferença entre a derivada calculada e a que deveria ser?
    
- estou passando primeiro uma L-BFGS antes do adam, mas ainda não entendi como os hiperparâmetros funcionam. às vezes só itera o tanto certo, às vezes um valor super alto (já fiz conta multiplicando pelo nº de camadas e neurônios e não bateu exatamente). Removo? Ajuda a convergir mais rápido
- Como fazer gridsearch nesse modelo? Faço um iterador (for's nested)?
    ==> Tem que fazer na marra
    ==> Desenvolver uma maneira de fazer otimização de hyperparams num loop externo fora da otimização dos params da rede
- Esses valores de erro que travam (fica sempre o mesmo erro na coluna) tem maneira melhor de "empurrar" ele do que aumentar steps? Weights? Ex: batch-fed tem um 1.36e-02 em 26k e 46k ele continua lá...
- O que são os vários valores de loss? No geral é 2x as variáveis de saída
    ==> Bacia de atração --> Mínimos locais muito largos, é isso que provavelmente
    seja o erro de estagnação. Resolver aumentando **epochs**.
        --> É estocástico, ele vai sair dali a partir de determinado sorteio.
- Citar biblioteca/linguagem?
    --> O problema de PINN efetivamente é na lib, cita ela
    --> código fonte pode ser publicado no artigo

## Problemas
- Batelada Alimentada o problema parece ser que ele não consegue resolver o volume direito e isso acaba impactando todo o resto. Vou botar um weight só pra ele e testar.
    ==> Adim e aumentar steps

## Comparações a se fazer
--> Comparar estratégias de adimensionalização
- Necessário fazer mais de um modelo (existem valores diferentes para concentrações iniciais diferentes) mas acho que nem se aplicam bem pq como dependem da conc. inicial, por exemplo, meu cstr é quase um "a título de curiosidade"
--> Comparar com euler e runge kutta (métodos numéricos) e dados experimentais do artigo
--> Já daria certo só no batelada : faz o pINN pros 3 e resolve+compara só batelada
- Medir tempo de execução entre cada um dos modelos (já pra resolver) mas tb expor tempo de "criação" do modelo


## Foco
<!-- - Aplicar PINN a bio-reações, em específico a solução da reação de produção do ácido lático pela bactéria Lactobacillus casey usando como substrato Lactose de Whey (Altiok 2006) -->
- Estratégias de adimensionalização para solução de cinética de bioreactions com PINN

## Título do Artigo?
<!-- - Physics-Informed Neural Network applied to Biological Reactions: A Lactic Acid Production case study -->
- Strategies of non dimensionalization in Physics-Informed Neural Network applied to the Kinetics of Biological Reactions: A case study with Lactobacillus casei
    --> To assumindo que a adimensionalização vai melhorar e resolver meus problemas. Mas ela vai mesmo????

## Sugestão de revistas
- Digital Chemical Engineering --> Opensource, valor 0 pra publicação até dia 31/mar
- Canadian Chemical Engieering (ou algo parecido)
- Process (É só de eng. química)
- Pedir pro Ivanildo :: revistas de bio tradicionais tlavez tenham apelo maior


### Programa
- Criar arquivo com .git
- Separar classes e funções em cada arquivo
- Um arquivo SÓ pro conjunto de dados do altiok já preenchido (não a classe em si)
    - Tem a opção get_all que retorna uma lista com todos
- A main simplesmente chama a função, tem 2:
    - uma de construir e plotar um modelo específico
    - uma de iterar várias configurações passadas em vários modelos
        - essa daqui acho que basicamente vai ser um gridsearch que recebe a lista de parâmetros
        para iterar
        - aí retorna todos os modelos e o index do de menor erro??
        - os modelos tb devem conter variáveis como os parâmetros que os geraram E o erro E o tempo de execução!

### Escrita
- separa mais referências de PINN, incluindo o artigo original
- 1 imagem explicando PINN, 1 explicando o sistema biológioc e o conjunto de reações que o representa
- Tenho um pouco de dificuldade na parte matemática mas vai sair. Acredito que quando ler mais uns 2 ou 3 artigos ande.


---------------------------------
- O meu modelo anterior independia de parâmetros de processo, bastava as concs. de entrada. Esse outro depende totalmente dos parâmetros. Comparo os 2? Deixo o outro pra outro artigo? Etc