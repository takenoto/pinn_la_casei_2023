# Todo List


## Main
  - Cronograma com datas em que pretendo concluir cada um desses trambei
  - Seguinte, como vou ter que justificar o número de epochs de treino, até isso vou ter que variar. Então é melhor fazer cada treino com HL fixo e aí depois junto tudo na mesma pasta...
  - #TODO eu já devia exportar os dados que quero agrupados. Imagina a trabalheira que vai ser ir abrindo os jsons um a um pra pegar lr vs hl vs nl???? Só se eu fizer um script que plota os gráficos do quye tiver numa pasta e aí eu rebolar os que quero dentro dessa pasta. É 8um pouco menos burro...
    - Pega a LR Ótima pra um Adam alto. Depois só fazer o teste reduzindo adam. Mas não tem cabimento fazer isso tudo junto. 504 testes, vai acabar nunca!

## Entregáveis
  - Reação:
    - Gráfico LR x Loss x NL. Um pra cada HL. totalizando 4 talvez então?
    - Determinar a menor loss que representa reação adequadamente
    - Fazer loss v4 e v5, comparar
    - Depois disso tudo => aí sim faça uma variação em um tempo discretizado maior e menor que o intervalo estudado de fato para ver no que dá...
  - Reator SÓ VOLUME
    - Predição APENAS do volume para batch, CSTR e 2 casos CR. Funciona? não preciso esticar aqui. Só pra validar ele individualmente e avaliar possíveis problemas na formulação...
  - Reator batch
    - Testes de LR, HL, NL, tempos de treino e predição (comparar loss 4 e 5) etc
    - Validação contra o gráfico experimental
  - Reator CR1
    - TODO
    - Seria mais interessante jáf azer ele parametrizado? É possível??

## Todo list - qualificação

- TODO eu tinha ficado de ver pq a derivada com o não lin dá tão diferente mesmo quando os valores das coisas em si batem em cima. to fazendo alguma conversão extra desnecessária na hora de plotar??? O None dá certo então deve ser isso... Meu método to_nondim tá errado???

- Faz o nondim como um dict externo
- E lá já deixa as opções salvas...

- verifique de novo o gráfico de derivadas. Ainda acho que tem algo nondim errado. Tem que testar pelo menos no batelada 1x com nd=1 (sem nondim) e outra com nondim linear pra confirmar que tá OK.

- Tá tudo ok. Queria treinar só de 30-60% do tempo mas propor condições em t=0. É possível? Pelas docs da deepxde parece que t=0 precisa necessariamente ser o tempo da IC...
- Veja "solve parametric pdes" => https://deepxde.readthedocs.io/en/latest/user/faq.html
  - Isso deixaria eu usar diferentes condições iniciais
  - Talvez uma geometry do tipo hypoercube conseguisse resolver
    - https://en.wikipedia.org/wiki/Hypercube
    - Aí eu conseguiria fazer um modelo generalista
    - Mas isso é só pra depois né. Primeiro vamos brincar de fazer o cstr funcionar...
- Rodar só a reação com loss v5 contra lossv6 já mostra que a 5 para a REAÇÃO.
- Ler o artigo que o Amaro enviou por email
- TODO Sabe o que era bom fazer? Um dos testes ser SÓ o volume de saída. Aí avalio só a reação, só o volume, e depois eles combinados.



- EU JÁ ANOTEI UM PEDAÇO. ACHE O RESTO E ANOTE TB.

- CSTR => Rodar um que inclua o antigo que prestou (09/09/2023) mas também faça um teste adicional pra nondim sem o dimensionamento pra 10.

* Quando for fazer testes pra medir performance treino/predição tem que marcar eles porque de forma geral sempre to fazendo outra coisa no computador enquanto roda então não é um teste justo.

* Parece que essa minha loss nova é BEM mais lerda que as antigas. Talvez por causa dos termos extras? A rede 60x5 tá uma lerdeza sem fim.

NL = [ 4, 8, 16, 32, 60, 80, 120]
HL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LR = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
funcs = [ tanh // swish // selu]

Não precisa variar tudo junto.
Por exemplo, pode só fazer um gráfico da LR pra avaliar quais produzem loss menores e faz um gráfico como (esse)[https://datascience.stackexchange.com/questions/63223/learning-rate-scheduler]

# Rode o cstr   

- Novos experimentos
  - Só reação, sem volume => volume vem do numérico => Batch com t e V de input
  - Batch normal, já prestou
  - CSTR => É de fato um CSTR, já inicia no volume máximo e só vamos ver as concentrações variando.
  - CR1 => CR que possui Vo = 1L
  - CR2 => CR que possui Vo = 0.1L
  - Esquecer o fed-batch????

- TODO estabelecer os limites da investigação. Número de neurônios máximo e mínimo, layers, etc.
- TODO veja se essas coisas tf.ones_like(X)*Xm não tão considerando a multiplicação separadamente e zerando o próprio X e afins
- TODO tá parecendo que essa condição de somar o X quando X<0 faz com que o erro tenda ao próprio valor de X pq a derivada é muito pequena, daí o erro vá zerando quando X = 0. então incentiva essa solução trivial.
  - A outra opção é fazer (loss_der + loss_minmax)
  - Pareceu resolver usando X^3 porque aí se ele for próximo de zero mas levemente negativo não vai puxar a solução na direção de uma solução trivial. O que imagino que aconteça: X < 0  deve ser tipo -0.2. dXdt uns 0.01. O X é muito mais relevante e é mais fácil simplesmente zerar tudo. Com X^3 o valor de X sendo relativamente baixo não vai perturbar tanto o aprendizado.
- TODO loss v4 => igual à v5 mas sem esse termo ao cubo. Deixe igual. E prove que ele faz muita diferença. Talvez?? Mas pra isso seria bom tb eu no numérico salvar os valores de dX/dt e afins. Porque aí posso comparar a ordem de grandeza com os XPSV e ver se minha hipótese tem alguma coerência?
- Talvez eu não precise esmiuçar 1 a 1 todos, principalmente o que não der certo. Isso me deixaria até mesmo fazer mais reatores se fosse o caso.
- Novas cores: laranja e marrom pra loss
- Testes: delineamentos
  - Descobrir quais eram os tempos adimensionais e refazer. Acho que a diferença é só que tem um So dividio pelos Ks+So
  - Nondim precisa ser testado em pelo menos 3 LRs diferentes pq talvez isso impacte significativamente nos resultados... Já que qualquer modificaçãozinha muda muita coisa e dificulta o aprendizado pra valores de predição menores. Talvez seja até por isso que adimensionalizar entre 0 e 10 tenha aparentemente mostrado resultados melhores.
  - Funções de ativação
    - Mantendo as mesmas camadas, treino, etc, testar:
      - Selu, swish, tanh
  - Teste o treinamento com decaimento. Talvez resolva minha vida ao evitar soluções triviais.
  - 

- Deixar tudo anotado no arquivo de cases_to_try
  - Começando pelos t adimensionais
- Listar forma de distribuição dos pontos de teste...
- Lista TODOS mestrado qualificação:
  - Descubra TB como salvar o gráfico de loss por epochs pq como foi feito tá uma merda preciso ficar voltando era bom ter outra forma, botar no json mesmo
  - Reduza variáveis. Tira teste weight e usar sempre a loss V3, ignorar as outras. Aí começa tudo de novo
  - Aí acho que devo fazer em metodologia uma seção de nomenclatura que explica que 80x5 é uma rede 80neuronios 5 camadas 3 que a outra nomenclatura é por exemplo 80-20-30
    - Teria como fazer também esses testes entre intermediários
  - TODO cheque o teste t_nondim e depois faça uma tabela com as coisas a serem testadas. daí simplesmente replique pra cada reator. Ponto.
    - E já com entrada de células e produto em 0, como Luciana sugeriu. Só substrato.
    - Aí acho que poderia até fazer já a v4 da loss, ignorar as outras e voltar a incluir a selu
- TODO seria bom se eu fizesse testes extrapolando o tempo de treino, pra ver se o sistema realmente aprendeu a "física" da coisa. Principalmente no fb.

- Esqueci a pg de agradecimentos
  - Orientadores, banca, fernando -> técnica

# Todo list
- Colocar losses no json de novo e testar
- Eu bem que poderia deixar rodando e enquanto isso já ir escrevendo os resultados do cstr que prestaram e fazendo essas modificações de metodologia viu...
- Pergunta no github: por que o display_every impacta nos valores salvos de loss e epochs? Isso não faz sentido. Deveria mexer tão somente nos prints que aparecem na tela. Tem 0 coerência mexer nos valores das List que são salvas no fim.

- Metodologia
  - Loss v1, v2 e v3
  - Nova nodim => Escrever no documento
  - Tabela 1 está com vários dados como "0" no lugar de ter um valor
  - Tabela 2 teve o mesmo problema. Talvez tenha sido quando troquei de computador? Não entendi, sinceramente.
  - Incluir o MSE na equação da loss. É o MSE qpra cada termo de cada vetor (XPSV) eu acho. Veja docs.

- Resultados
  - Fed-batch: new random tests with high n_domain and n_test to find at least some acceptable range to work within
  -
  + a 
  - Refazer gráfico 32x5, o primeiro da seção de conclusões
