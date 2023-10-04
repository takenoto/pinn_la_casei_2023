# Log

## Testes a se fazer

- FED-BATCH
  - Novo tempo de simulação, menor...
- CSTR
  - As melhores redes do batch...
  - Loss v4 com os 3 nondims tanh 60k adam
  - Tentar scaler 1/10 pra todos mas 1 pro volume. 10 pro tempo vai cancelar então tb daria certo...
  - SGD x Adam e tanh x swish
- Tente abrir as redes que deram flat. Tacou um bias? Imagino que se valores ficarem entre 0 e 1 ele possa tb estar ignorando muitos dos bias, zerando, ou o contrário, ignorando os inputs. Isso é uma info legal embora eu não saiba bem interpretar ela.


## Anotações
- Loss v4
  - Batch
    - Já fiz vários nd com t e sem t e linear e vi que muitos ficam ok.; A loss v4 parece estimular o volume a ficar caindo e não oscilando.
    - None e Desvio validados! Desvio não ficou tão bom mas pode ter a ver com a minha escolha de scaler pras variáveis.
    - Batch
      - scaler 1/10 pra tudo pareceu melhorar resultados nondim linear
  - Adimensionalização + scaling pra deixar todo mundo entre 0 e 10 parece a melhor opção até agora. Excluindo sempre o tempo da adimensionalização, claro... Será que fica bom entre 0 e 10 justamente por ser o mesmo range de variação de t_final? Que vai até 10.2...
  - É, mas também foi o único em que eu botei pra ir até 55k otimização. Isso deve contar mais que todo o resto zzzzzzzz.... Os outros eram só 35k eu acho.


## by date

TODOS: t2-f1d10 mas na rede 30NL, aí comparo com a 20...
- Eu acho que aquele teste que tinha ficado bom foi praquele reator que a variação de volume era quase nada e que eu tava tentando ver a partir de quanto a simulação desandava e beirava o impossível de treinar.
TODO: testa alguma rede maior tipo 80x5, aí nem precisa fazer micro variações de outras coisas. Só pra ver se sai algo que preste mesmo.

### 2023-10-04

- Rede 20NL, lossv5:
  - t2d10-F1d10 (em progresso)

### 2023-10-03

- Novos resultados: (30 NL por padrão)
  - reactor-V0-3--Vmax-5--Fin-25E-2
    - t2d5-1
    - t2d10-1
    - t2d100-1
      - ^ Esse foi de longe o pior. Os outros não teve um que foi efetivamente melhor que os demais...
    - t2d10-F1
      - se ficar ruim, talvez eu tenha que fazer na rede 20x de novo mds... Agora pq funciona num e na outra não? 0 sentidos viu
    - t2d10-F1d10
      - A partir desse, incluí o teste pra LR=1e-2 porque era a usada no original, que tinha dado certo...
      - Esse antigamente tinha dado certo, e motivou essa escolha: ND-Lin-t4-F1d10 in_t out_XPSV tanh tr-0-25pa L5 LR-E-2_1 20x2 p16-32-32 45kep lbfgs-1 TD-Hammersley m-
      - A única diferença é que eram 20 neurons e não 30. Pelo amor...
    - t2-F1d10 SÓ 20 NEURONS X2-3-4
      - Não ficou bom. Refazer com t2d10 pra 20NL.
      - opinião ????

### 2023-10-02

- Novos resultados (t2x100, etc...)

### 2023-09-29

- Mais testes com variação de nondim  (t2, t2x5, t2x10)

### 2023-09-28

- Criação de modelos t2 x10 x100 x5 e d5 d10 d100;
- Voltei à loss v5. A v6 é muito cheia de lero lero e dá muito trabalho.
- Me parece que ainda assim são MUITO sensíveis à LR... Mesmo com rede 30x3. Talvez fosse bom eu testar em combinação com os F1d10 e afins (só fiz com F1 e os t2 x e d valores). Em progresso porque demora muito. As redes 10x não estavam tão boas então decidir aumentar pra garantir que não fosse por falta de capacidade da rede... Esses testes todos foram só 0-15pa. Alguns quiseram prestar mas prestar que é bom nenhum rolou.
- De novo tem uns ok mas desandam depois dos 15% de tempo. Acho bom eu fazer um 0-30pa e daí já testar. Pego as redes 10 e 30 x2 e x3 e daí rodo pra vários LR uma ou duas nondims por vez
- Mais testes com redes 30x3 e afins pra NDN t2 e t1. Os outros não consigo fazer hoje, vão ficar pra amanhã. De vez em quando tem uns interessantes, mas francamente nada muito bom.

### 2023-09-26

- Modelos t6F1d10 para CR... testando.

### 2023-09-25

- Novas variações da loss v6
  - Multiplicar loss dVdt por 50 resolveu mas ainda não tá muito legal.
- Consegui alguns resultados aparentemente aceitáveis usando t6 e Fd10 com tempo 0-15pa para o CR menorzinho. Já é um
- Loss v6: nova iteração zzzz... Estava dando default pra loss v5 pra tudo com exceção de V, por isso minha modificações não tinham surtido efeito algum.
- Voltar a fazer treinos na região 0-15pa e ir ajustando a loss na marra com base em hipóteses, e só depois ir pras demais...

### 2023-09-22

- Implementar variação de CRs com params
- Nova lógica para geração da id do modelo
- Testes preliminares com redes minúsculas (2-10 neurônios, 1-3 camadas) no CSTR. 0-20 pa mas tá tudo indo relativamente bem. O problema realmente devem ser nos valores com pico e queda...

### 2023-09-21

- Nos gráficos 3D a escala log é bugada, então vou ter que fazer por fora se for o caso. Colocar log_scale faz somente com que exiba os número em log (10^-3 etc) mas não vai colocar em escala log de fato...

### 2023-09-20

- Add garbage collector
- Add novo estudo de caso CR: cr-4x5L. O reator inicia em 4.5L e só vai até 5L. Assim consigo ver se numa variação pequena ele ainda teria essa dificuldade que os outros CRs (fora o constante) estão apresentando. Deu pra ver que não, ele fica bem tranquilo e dá tudo certo rapidinho. Isso serviu pra validar o equacionamento...
- Novos testes pros reatores CR. Não preciso botar todos mas quero ao menos ver se chego em alguma conclusão razoável.
- Pelos meus testes do reator cr-1E-1L o problema não parece ser meramente a profundidade ou neurônios da rede, porque a 80x3 também deu na mesma. Realmente acho que é algo no treino ou na loss, e que talvez eu arrume ainda nas redes 10x. Como tá demorando MUITO. Acho que vou ter que fazer cada NL, HL e Nondim individualmente por todos os LRs. Se não for assim vai passar de 60 iterações, e a partir daí elas são tão absurdamente mais lentas que simplesmente não compensa.
  - Olha, pensando bem, eu poderia começar fazer só 0-10% do tempo de simulação, depois 0-20% e indo assim até encontrar qual o "t" problemático, se é onde acontecem os picos...
- Mais testes pro CR. Precisei aumentar adam pochs pra 120k e ainda assim não sei se foi o suficiente. Testei apenas apra t7  FD Lin.

### 2023-09-19

- Novos testes com nondim e redes 30x e 10x. A adimensionalização das variáveis sozinha (F1d10) foi consideravelmente melhor que as outras e que as opções sem adimensionalização.
- Testes nondim t
  - Testes em rede 10x3 adimensionalizando o tempo e XPS
  - Testes em rede 10x3 adimensionalizando apenas o tempo
  - Adimensionalização t4 parece ser, no geral, pior que as demais (é a /10)
  - t only
    - t7 ficou bom em 3 faixas de lr bem distintas (1e-2, 1e-3 e 1e-4). Isso é muito bom.
    - t3 ficou aceitável em 3
    - vários outros ficaram bons em 2 das 3 faixas, incluindo o sem nondim. Eu iria de t7. Mas o valor numérico de t7 é bem próximo do t2, então seja só uma coincidência.
- Add param: solver params > is_save => determina se vai salvar ou não o modelo, padrão é falso
- Bug fix: o que foi salvo como dV/dt era na verdade dSdt. Só não deu em nada porque todos os testes até agora tinham sido somente a reação, então não tinha dV/dt para ser salvo. Código estava em pinn_saver.py.
- Implementar: Agora é possível rodar apenas a predição de volume
- plot_comparer_multiple_grid => Agora seta automaticamente limites superiores e inferiores de y para cropar valores muito próximos e evitar aquela notação "+5" no topo, por exemplo, quando os valores variavam entre 4.9999 e 5.00001.

### 2023-09-18

- Finalizar testes redes p/ reaction only
- Até mesmo a rede 10x2 teve uns casos aceitáveis. Vou então fazer o teste variando a LR.
- Criado o arquivo json_viewer, que abre todos os jsons dada uma lista de nomes de arquivos. Daí ele pega as informações marcadas (através de um callback para ser chamado por fora) e fecha os arquivos. Assim posso pegar por exemplo todas as últimas loss dos arquivos X e Y.
- Plot caller e gráficos relacionados + integração com o json_viewer
  - Padronização do dict para scatter
  - Gráfico scatter implementado
  - FIX: Figura era resetada ao girar, o que salvava uma imagem em branco.
  - UPDATE: Mudado o padrão do dict passado por argumento do 3d plot.
  - Plot por ângulo funcionando OK
  - 

### 2023-09-17

- Novos testes com LR
- Determinação que treino por Hammersley é bem melhor que distribuição uniform. Ver: "results\2023-09 teste preliminares pt2\reaction Altiok\1 hammersley e uniform dif ABSURDA"
- Corrigido: em "grid_search.py" File não estava sendo close() (fechado) o que impactava a performance e esculhambava o arquivo pinns.json
- Testes de NL e LR para reação. Fiz 1/6.

### 2023-09-16

- Corrigido bug no plot da derivada. Era pra ser a derivada normal, e não a adimensional mesmo.
- Cases to try: substituído forma de declarar nondim scalers.
- Implementação explícita das condições de adimensionalização do tempo e demais variáveis
- Testes para reação: LR e Epochs
- Testes de LR e Adams. Fazer todos juntos fica inviável pelo tempo eu acho...
- Mais testes de noite (19-22)
  - De 0-60% t na reação only batch fica bem fácil de acertar. 0-90 desanda.
  - No gráfico: agora plota uma linha indicando o intervalo de treino.
- Criar arquivo "thinking.md"
- Encontrei algumas redes boas em 30x3 para 0-90pa e com t_ND=caso "t2". 
- Pra amanhã, é bom ver se consigo desenrolar logo e dar uma olhada nos testes do t nondim...

### 2023-09-15

- Permitir (cases_to_try, grid_search, run_reactor, solver_params json) o uso de tempos min. e máximo do experimento diferentes de 0 e t_max para avaliar isso.
- Fazer o reator batelada num tempo maior como uma configuração extra e ver como ele se comporta, se também fica com erro alto, etc.
- Daqui pra frente acho que mini batch pode ser totalmente ignorada, vamos trabalhar só na distribuição de pontos e ver o comportamento dali.
- Parece não ser uma solução trivial (do ponto de vista matemático) mas do ponto de vista de ML: zerar X e botar tudo constante era algo que baixava muito o erro sem ter muitas consequências. 
- Removi o termo que multiplica a loss minmax  por 10. Como o valor absoluto já é maior que a derivada, multiplicar ele faz com que a derivada fique menos relevante para a composição da loss e talvez isso implique naqueles erros. Além de que ficou mais difícil o processo de treino.
- Corrigido: plot de derivadas agora é salvo com o valor adimensional para poder ver a sobreposição independentemente da escala e ficar mais fácil de comparar. Foi possível ver claramente a predição errando o volume e acertando o resto. Mas aí "acerta" o resto já descontando o erro do volume, então também erra!
- No json de pinn => adicionado também as predições das derivadas
- Rodar só a reação com loss v5 contra lossv6 já mostra que a 5 é muito melhor. A 6 hoje, enquanto escrevo isso, é a mesma coisa que a 4 antiga. Então isso já valida a complexidade de ter um termo ^3.

### 2023-09-14

- Rascunho loss v6 multiplicando o erro<0 por 10 e por 100
- Adicionar valores calculados de d(X,P,S,V)_dt aos resultados numéricos
- Plotar por padrão, além de XPSV, d(X,P,S,V)_dt numérico
- Plotar dXPSV/dt do Pin... E comparar com o numérico. Já vi que estão indo em direções opostas no início do treino...
- Agora plotar o gráfico com t de simulação (numérico) vs t que foi discretizado (pontos não linhas) Importante pra eu saber se uma região está sendo muito desprivilegiada. Na verdade é uma linha de t e os pontos por cima pra eu poder comparar


### 2023-09-13

- Testar mais vários valores de loss, todos ruins. Implementei variação de LR. Estou preparando pra testar variar a loss mesmo, de novo zzzz.

### 2023-09-12

- Implement Autosave for each pinn using a callback
  - A 56 pinns simulation failed (OOM - out of memory) so all models were lost. This is to prevent it to happen again.
- Remove plot loss for multiple pinn. Implement individual loss ploting.



### 2023-09-09

- Loss v5 foi modificada, e sem a normalização automática. Resultados absurdamente melhores, praticamente todas as redes ficaram boas e com baixa variação, até mesmo do volume. Essa versão já pode ser a loss final porque acabou de ser validade pro reator batelada.
- Loss v5 validada com batelada.
- CR => Novos modelos definidos no código e no arquivo naming.
- Testei distribuições e salvei (Hammersley e LHS) mas não parece ser aí o problema
- Mesmo a versão CSTR puro (já inicia no volume máximo) fica bem péssimo. O volume até prediz ok, o resto desanda. Continua me parecendo que esse pico da função é super responsável por isso... O batch já validou o equacionamento da reação... Então isso tá certo, já foi confirmado. Ficou super bom mesmo nessa lossv5.
- Cheguei a testar 4000 pontos teste e domain + mini-batch de 20, continuou bem ruim.
- Um dos testes do CSTR finalmente funcionou bem. NN 16x1 com NDLin escalado a 10.


### 2023-09-08

Loss v5 // loss 5 // lossv5 // loss5 => As losses são normalizadas em 50% da soma para ficarem todas na mesma ordem de grandeza
  - L_n' = 0.8\*L_n + 0.2\*Loss_pde_total
    - Repare que isso torna a Loss total artificalmente maior, então não dá pra usar só ela pra comparar.
- Aumentar resolução dos gráficos dew XPSV e Loss (600 dpi)
- CSTR
  - CSTR Renomeado para cr("continuous_reactor")
  - Implementar novas condições do reator contínuo
    - X_in e P_in = 0
    - P0 não é 0 porque considera que foi incubado então já pode ter produzido algo
- Mudar cores loss train x teste em comparação com pinn vs euler, não faz sentido serem a mesma
- Ajustar limites dos gráficos pra não ficar aquele 4.99 + 0.0001 

### 2023-09-03

- Escrever novas nondim no documento (OK)

### 2023-09-02

- Erro no tempo do fed-batch. No texto está escrito 1036 (um erro porque deveria ser 10.6) porque digitei errado no teclado numérico.
- Escrever novas *losses* no documento e nos slides.

### 2023-08-22

- Encontrei um erro sério
  - Ao pegar o reator, sempre convertia como se tivesse adimensionalizado. Mas acontece que o método numérico nunca tava sendo adimensionalizado. I.E. os valores nunca são os nondim, mas os valores absolutos (NO NUMÉRICO). Isso significa que tudo de lá deve ser convertido para nondim.
- Adicionei o salvamento dos valores calculados numericamente, predição pinn e predição pinn ainda nondim, sem converter. Pro caso de precisar no futuro. Não é possível, agora tá fácil de mexer. Pelo amor...

### 2023-08-23
- Novo teste loss v4 + nondim batch: 6. Deu certo. O único problema é o volume caindo ao longo do tempo.