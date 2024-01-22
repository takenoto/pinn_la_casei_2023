# Todo List

- TODO Metodologia
  - TODO começa a corrigir metodologia. Ver coisas que precisam ser traduzidas, novas ilustrações e anota aqui.
  - TODO tem tanto as sugestões delas quanto o que mudei e preciso anotar lá
  - TODO refaz aquela figura explicando o PINN na metodologia
  - Preciso colocar a explicação de como foi feita a otimização da rede
- TODO Ler todas as anotações deles na lista de TODOs do mestrado e ir colocando aqui as que se aplicam
- TODO resultados
- TODO reescrever RESUMO e ver correções delas
  - TODO reescrever Abstract
  - TODO verificar gramática EN em pelo menos 3 sites
- TODO Introdução
- TODO Objetivos

-----------------------------

1) O CR de vazão maior funcionou. As redes da "primeira passada" foram 30x3 e 40x4, então é só olhar nelas pra pegar os gráficos de LR vs erro por exemplo.
2) Para cada caso, no trabalho, criar uma tabela com os valores dos scalers nondim já calculados e não só a referência à variável. Algunsdão 18, outros 10, etc, isso pode ser importante também.

------------ OLD

- os gráficos batch NÃO ESTÃO MAIS EXIBINDO OS PONTOS DE XPDATA. VOLTAMOS À ESTACA ZERO? SÓ EXIBE OS QUE FORAM USADOS NO ÚLTIMO, É ISSO???
- e tb pode tirar aquelas loss de bounadery e afins
- repetir esses testes de cima pra 3, 4, 5, 6, 7, 9, 10 neurônios por camada. Nisso já tenho gráficos pra praticamente tudo.
Vou ter NL de 3-10 e HL de 1-6
Acho que ficariam bem bonitos os gráficos XPSV por HL. Tipo pego 8 neurons e ploto em 3D lado a lado os pontos. Sò talvez fique ruim de ler e aí seria melhor o erro?

NLs: 
3, 4
5, 6
7
9
10

- Acho que nem preciso ir apagando. vai ficando tudo por cima e acumulando pra se precisar já ter feito.
- E aí só no máximo copio e colo numa pasta os de interesse. Mas até isso talvez seja desnecessário, porque acabo duplicando testes desnecessariamente e vai ficar cada vez mais difícil de achar onde tá cada um.

- Enquanto isso asssitir e ler artigos

1) Terminar outros valores do batch 8x
  -  Faltam camadas 1-4
  -  Faça gráfico LR x HL x MAD
  -  Depois LR x HL x erro absoluto %
    - Aí faz gráfico triplo: erro de N, dNdt e da derivada 2ª
      - 6 figuras: 3 de erro % absoluto e 3 de log(erro absoluto)
2) Faz o processo novamente pra
  1) Sem nondim
  2) Nondim com Lin-UPx1 ao invés de Lin-lin
  3) O melhor dos anteriores, mas trocar a loss pela A1 normal.

## Ler papers
### 1) DeepOnet
https://arxiv.org/abs/1910.03193
### 2) os que ficaram no celular e o que eles me mandaram que li mas faltou olhar a fundo

- No desenho do fluxorama que vai representar o processo, incluir a etapa de limpar objetos da memória
- Talvez testar a função de inicialização seja ainda mais importante e eu esteja negligenciando
https://stats.stackexchange.com/questions/229669/when-should-i-use-the-normal-distribution-or-the-uniform-distribution-when-using

- Referências, add: https://colorbrewer2.org/#type=diverging&scheme=PuOr&n=3 
Colorbrewer na escolha das cores

Loss v8: é a 7 mas sem o abs que ajusta o sinal. Parece que ele faz com que a loss represente melhor o desvio (errou = loss alta) e que evita as zonas de solução tivial. E mesmo quando vai pra elas, o erro fica alto, o que é muito justificado. Então como um todo ela é mais coerente.

- Testar nova rede: só uma HL!!!!!!!!!!!
- Releia artigos pinns e veja a metodologia que já esqueci zzz

# Gráficos

- Colocar os ticks como opcional a depender dos parâmetros passados
- Colocar a fonte de matemática lá no arquivo base como a mesma pra notação científica ????
- Agora que já foi feito muda a fonte de mat no arquivo de configuração e isso deve resolver tudo
- usar locator_params para setar nbins =4 pra todas as direções. Assim só vão ser 4 ticks e não vai ficar feioso.
  - depois pega como argumento e só usa 4 como default
  - https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks
  - https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.locator_params.html


Testes com CR:
- Atualizar a loss v6. Colocar o peso do volume enorme pra tentar punir essa ida a 0 no começo, que é o que tá esculhambando tudo.

GRÁFICOS:
1) Colocar uma surface
- https://aleksandarhaber.com/explanation-of-pythons-meshgrid-function-numpy-and-3d-plotting-in-python/

Já que eu to é eguando com esses testes, seria bom já agilizar logo o da variação de volume apenas...
1) Anotar em thinking meu passo a passo e seguir ele rigorosamente pra eu poder terminar...

1) Como o contorno tá nondim a loss  dele dá artificialmente baixa e isso prejudica tudo. Talvez eu pudesse colocar weigth x10 pros pesos???
2) Por isso que o d10 fica bom, porque é como se eu tivesse botando um peso de 10x nelas zzz...
)1º Comparar F1 x10 e d10 e daí já escolho só 1 dos 2 pra seguir em frente.

0) Exibir todas as legends não só umas
1) Reduzir tamanho do texto das legendas e padding. Tá ocupando espaço demais.
1) Implementar surface plot também
2) Fazer plotar vários num só possível passando cols e rows...
3) Encontrar ângulos melhores que escolhi uns meio feios viu

1) Testar se salvar funciona e plota todas as variações de ângulos...
2) Botar estilos como cores e afins
3) Colocar color bar caso se aplique?????? veja os antigos que fiz

2) Gráfico 3D surface. Como? acho que os 2 são 1 só e muda só o dict que passo, viu...
https://stackoverflow.com/questions/12423601/simplest-way-to-plot-3d-surface-given-3d-points
https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib

- Adaptative LR:
https://deepxde.readthedocs.io/en/latest/modules/deepxde.html?highlight=decay#deepxde.model.Model.compile

GRÁFICOS
0) Acho que eu poderia fazer pra 10x3: Loss vs ts vs escala dos outros (0a1, 0a10, 0a0.1 etc)
1) Fixa por rede. Ex: 10x2. Aí faz o plot adam vs LR por MAD no lugar de loss e outro com loss.
Além disso, faz com e sem lbfgs. Aí passo a ter respostas que prestem.
JUMENTO SE TU FAZ O ADAM A 80K TU JÁ FEZ TODOS OS VALORES INTERMEDIÁRIOS WTF TÁ FAZENDO TUDO 200X ??????????????
Talvez fosse bom fazer um gráfico que fosse a diferença entre LoV e LoT. Aí consigo ver os que tão divergindo???
2) Talvez ficasse legal X por LR e tempo. Aí daria pra ver justo os que divergem...

# IMPORTANTE
- Talvez eu devesse não plotar o resultados de todas as redes nas mesmas settings, mas o melhor resultado de cada uma? Dentre todos os investigados???

Minha sugestão de metodologia por enquanto tá assim:
1) Teste aberto geral para ter noção e ver se alguma rede já consegue resultados minimamente razoáveis
2) Explorar adimensionalização na menor rede testada e na melhor, já com variação de LR

# Parte 2 : Reator
- Faz investigação dos 3 modelos com variação de volume pra ver se tem bode em algum canto.

- Já que a rede 100x deu tão certo, fixa uma delas e varia LR e Adam. Aí vejo a influência deles.

1) Fazer Hammersley reaction loss5 => faz a distribuição de pontos 1 a 1 pra ver qual é melhor. São 96 ao todo então fica demais fazer tudo mas se for por partes consigo terminar ainda hoje...
  1) OK
  2) OK
  3) faltando
  4) faltando

## Main
  - Cronograma com datas em que pretendo concluir cada um desses trambei
  - Seguinte, como vou ter que justificar o número de epochs de treino, até isso vou ter que variar. Então é melhor fazer cada treino com HL fixo e aí depois junto tudo na mesma pasta...
  - #TODO eu já devia exportar os dados que quero agrupados. Imagina a trabalheira que vai ser ir abrindo os jsons um a um pra pegar lr vs hl vs nl???? Só se eu fizer um script que plota os gráficos do quye tiver numa pasta e aí eu rebolar os que quero dentro dessa pasta. É 8um pouco menos burro...
    - Pega a LR Ótima pra um Adam alto. Depois só fazer o teste reduzindo adam. Mas não tem cabimento fazer isso tudo junto. 504 testes, vai acabar nunca!

## Entregáveis
  - Reação:
    - Fazer loss v4 e v5, comparar
    - Justificar observações de n_points e afins. Pelo teste inicial já parece que só se justifica em layers muito pequenas (16x3) as 30x2 e 30x3 não parecem precisar muito.
    - Gráfico LR x Loss x NL. Um pra cada HL. totalizando 4 talvez então?
    - Determinar a menor loss que representa reação adequadamente
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
