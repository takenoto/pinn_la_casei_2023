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