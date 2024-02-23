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

### 2024-02-19

- Iniciar correções da metodologia no arquivo.

### 2024-01-28

- TTODO

### 2024-01-22

- Rodar o CR t9F1d0 mas variando o npoints ao invés de LR e redes. Faz vários npoins pra 30x3 e 40x4.
  - p(8, 8, 8)
  - p(3, 32, 32)
  - p(8, 32, 32)
  - p(8, 300, 300)
  - p(300, 300, 300)
  - p(8, 1200, 1200)
- Fazer dados para obter rede somente para a cinética (não tem V como output e V=1!) para 30x3 e 40x4 variando LR
  - t9F1d0 não pode porque Fin precisa ser != 0
  - t7F1d0 e t1 sem nondim 52.624265469999955 min
  - t1F1d0 e t1F1x10  84.85971691000013 min
  - t8 e t6 F1d10 66.99975563499999 min
- [EM PROGRESSO] Ilustrações para metodologia
- TODO faz gráfico com imshow => pontos inicias vs pontos testes e treino vs erro
- TODO a conclusão parece que será a mesma de antes: usar nondim viabiliza uma quantidade maior de redes e de LRs, então supostamente ficaria mais fácil encontrar um bom conjunto de hiperparâmetros simplesmente porque existem mais possibilidades. Algumas redes sem nondim ficam até boas, mas as "do lado" variando só um pouco LR, HL ou NL já ficam péssimas ou dão aquela solução zerada que é ainda pior que uma solução trivial, porque a solução trivial pelo menos é válida.

### 2024-01-18 + 2024-01-19 + 2024-01-21

- TODO rodar pra p8-32-32, pra montar gráficos: (deixei redes NL40 e 30 reservadas pros testes primários e fim)
  - Rodar tempos:
      - t1-1 (sem nondim) pra comparar a performance em redes minúsculas : 118.554348555 min
      - t9 132.62632637333328 min
        - OK uma vez pro wAutic, pra poder plotar os 2 e comparar
        - TODO uma vez pro wA1
      - t8 ~ 112.72365334333332 min
      - t7 ~ 117.15610333333328 min
      - t6 ~ 146.886468555 min
  - t9F1d10 para alta vazão e 0-50pa para ver se funciona com mais tempo || t~95.40187755666666 min
  - [ABANDONADO] t9F1d10 para o reator de baixa vazão e 0-25pa pra ver se presta
    - NL: 4, 10, 20, 35, 60, 80,
    - HL: 2, 3, 4, 5, 6, 8
    - LR: 1e-3, que foi a do menor erro: "Lin-t9-F1d10 Lin-t9-F1d1040x4 tanh L7B LR-E-3_1 wautic-e2 Lin-Lin p8-32-32 45kep lbfgs-0-1 m2000 ic-2k"
    - wautic-e2
    - F1d10
    - 0-25pa
    - tempos: t9 (menor erro em um caso), t8, t7 e t6. Tá uma dominância BEM ÓBVIA desses 4 tempos. valores de t_nondim aproximado:
      - t6 = 3.77
      - t7 = 12.04
      - t8 = 18.87
      - t9 = 10.0

### 2024-01-17

- 🆗 pegar mais resultados ainda pra rede 30x3 e 40x4: com t2~t7 e F1d10, pra validar adimensionaliação do tempo.

### 2024-01-16

- 🆗 Fazer um com vazão mais alta (5e-2 L/s) pra ver como fica. De repente tanto é melhor enquanto processo quanto mais fácil de resolver. 
  - Ficou bom
- 🆗 0-15pa já sei que presta, voutestar agora 0-25
  - Pronto! 0-25 ficou perfeito! Basta simplesmente ver os que se saem melhores e acabou.
  - Rodei nessa ordem: Loop de fora nondim, loop interno n pontos (p8-32-32 etc). Já roda tudo com A1 x autic2 pra comparar também. Deu 36 por rede. Só vou fazer a rede 30x3 por enquanto mesmo.
    - nondims: 1 por vez, na ordem que estão. t8 demorou ~70 min. t1 demorou 78min, mas já estava assistindo. t9F1 72 min. t9F1d10 69 minutos e ficou MUITO bom.
      - Nº pontos, 1 por vez, na ordem em que estão.
  - OBS: sem nondim já pareceu ter ficado significativamente melhor
    - t1 => a 40x4 tem um autic vs A1 que fica bem melhor no autic.
  - Aí depois disso pegar os 2 melhores LRs, o melhor entreu autic e A1, e simplesmente faz várias redes pra ter um gráfico. Pode ser no p8-32-32 pra ser mais rápido.

### 2024-01-15

- OK ler vários dados no plot_caller a partir de uma pasta e agrupar eles. Eu basicamente copio os json da pasta do batch ou algo do gênero.
- OK agrupar dados
- Fiz alguns testes com o modelo sugerido pela Luciana (sem Xin, com X0) e ele também ficou bem ruim. Entãoa o invés de tentar tanta marmota vou aumentar os pontos e tentar achar um que preste.


### 2024-01-14

- Consegui implementar um imshow, já com exemplos e funcionando. Já vai funcionar bem pro que queria. Agora é testar e ver se fica legível com os dados que tenho.

### 2024-01-10 

- Reler código de plot_3D e plot_caller pra me familiarizar novamente com o que precisava ser feito.
- fazer um heatmap? Acho que é mais fácil fazer por fora mesmo não? Sei lá

### 2023-11-28

- Rever código da plotagem de gráfico...

### 2023-10-25

- TODO separar TODOs anteriores e ou deleta ou faz deleta fim
- TODO faz pelo menos um gráfico 3D pra ver se minha ideia original funcionou...
- TODO tenho que fazer é um heatmap zzzz
- TODO ver escopo do trabalho e já deixa uma lista com NL e HL que foram testados, e afins, pra comparar nesse heatmap já sabendo o que será.

### 2023-10-19

- disregard_previous_best=True no solver adam
- testes modificação equação de balanço dNdt no PINN
- Erro numérico: dXdt estava apenas dX, não dividia por dt...
- Fiz o teste de calcular como se fosse em mols a variação, não funcionou
- Não estou sabendo incluir o efeito de diluição
- FEITO:: troca tudo e conc pra número de moles
  - ODE preparer
  - run_reactor
  - loss v7
  - Do jeito que fiz a conc. vai estar sempre errado em V=0 (aliás, em V < V_threshold)... Porque to pegando o número de moles e aí vai pro infinito...
  - nesse ritmo vou ter que trocar a adimensionalização porque são valores em concentração... E não faz sentido essa "adimensionalização". Talvez eu precise fazer o que?
  - o grande problema é que, mesmo que não acontecesse nada, só a diluição deveria ser contada já. E não vejo ela aparecendo em nenhum canto. Então não tem como o efeito dela ser contado...
  -  desfaz tudo isso aí e faz o efeito de diluição, que é o que tá faltando...
- FIXED: O equacionamento do PINN não estava contabilizando o efeito da diluição:
  - previous: dNdt_calc = rN + (f_in * inletN - f_out * concN) / V_th
  - current: dNdt_calc = (f_in-f_out)*concN/V_th  + rN + (f_in * inletN - f_out * concN) / V_th
- Fiz pra variação de volume com vários pontos de treino diferentes e em todos ficou volume bom e querendo zerar X, de novo, pra zerar a loss mais facilmente.
  - Vários pontos 8-32-32, 20-300-300 e 40-1200-1200 e todas deram errado.
  - Mudei camadas também e não deu em nada. Então pelo visto tenho que ir por outro caminho.
  - O problema continua sendo essa tendência infeliz em zerar o X.

- Todos só 0-50pa:
  - Cada um desses testes faz pro CR que já inicia no máximo e na curva suave:
    - Lin-Lin-t1-1, Lin-Lin-t2-F1, Lin-Lin-t2-F1d10
    - Depois UPx1 F1d10 de t3 a t9
    - Daí já devo saber se uma dessas formas de nondim é melhor que as demais comparando, por exemplo, o perfil que ultrapassa a região de treino.
  - TODO: veja se pro cstr sem variação de volume, contínuo do início ao fim, ele fica correto...
      - E com Xo e Xin = 0
  - Xo e Xin normais (Xin sempre é 0 mas enfim)

# TODO ficaram faltando esses dois 2 aí: pro XPS do batch
        ("t1", "1", "Lin", "Lin"),
        ("t2", "F1", "Lin", "Lin"),
        ("t2", "F1d10", "Lin", "Lin"),
        em (8, 6),
            (30, 3),


# De verdade, antes de começar, organize os dos dias anteriores. Já tá gigantesco.

Testes: (obs: todos foram feitos enquanto mexi no pc, então train_time e pred_time não deve ser levados em conta)
1) TODO testar 3 variações de L7 em 4 tipos de reator (tudo padrão, tanh, etc), sem nondim, rede 10x3, LR1e-3:
  - OBS: todas as redes 10x3, e só com tanh, pontos 20-20-20. Depois olho outras redes e afins. Já sei que a 10x3 presta então vamo nela. Veja que são muitas variações de L7 então demora MUITOOOOOOOOO nem vale a pena. Então vou fazer só a 7G.
  - Rever: Olha, acho que a ordem de testes de nondim deve ser impacto do t => impacto de F1 e d10 e x10 já nos melhores ts sinceramente, poupa muito trabalho
  1º Roda os LINLIN LINUPX1  e No nodim simples
  2º roda os testes que exploram todo o tempo??
  - 1.1) TODO- Rede t => XPS :: Foi uma péssima ideia rodar tudo junto. Demorando demais, vai terminar nunca zzzz... A memória continua se acumulando por causa do python, mesmo com tudo mds. Chegou a 91% de memória em 200 iterações ainda do primeiro batch, o segundo nem ia conseguir rodar. Deixei a noite toda rodando, mais de 8h... Então vou trocar tudo, até a rede padrão. Só o 7A ficou bom em algumas dessas. F, G, H e I ficaram bons pro volume e ruim pras outras. F foi a melhor. Então farei uma híbrida.
    - batch (11, "Xo", "Po", "So")
    - batch (24, "Xo", "Po", "So")
    - TODO default => HORRÍVEL ENTÃO PULEI PRO PARTE 1
    - TODO parte1
    - TODO parte 2
    - TODO escolhe um único caso e faz todas as variações de 7 só pra mostrar que fica boa.
    - TODO comecei fazendo um pedaço da parte 3 que deve ficar melhor pra já me dar uma ajuda zzzz
  - 1.2) TODO Rede t => V
    - CR (1, 5, "1", "-4"), baixa vazão quase nada
      - Ainda não terminou mas no geral a "7G" parece ser a melhor. C e H também ok em alguns pontos.
      TODO 1º rode os 2 blocos default e 1, depois os 2.coisas
    - CR (1, 5, "25", "-2"), vazão normal
- TODO bota esses 2 testes anteriores já na pasta de testes definitivos, acho que rola
- TODO tentar abrir e comprarar MADs no plot 3D. Preciso saber se o modelo do json está certo antes de seguir em frente...
2) TODO Para o reator batch testar:
  - Contra todas as loss v7, uns 10 tipos de nondim (já inclui t escalado e não escalado...)
  - Nas melhores 3 ou 4 loss testar diferença de vários LRs e npoints. Isso já me diz se faz sentido aumentar ou diminuir ambos.
3) TODO Faz modelos t=>XPSV para CR com variação baixa (4.5L a 5L) porque é capaz de funcionar e validar. Depois que vou pros de 1 a 5.
4) TODO Faz Modelo batch com hypercube entrando tudo. Esse vai ser bem mais trabalhoso pq vou ter que mexer em cases_to_try, run_reactor e no pinn_saver pra usar as condições. E pode me salvar se o modelo de (3) acabar não prestando.

 -T-O-D-O veja se consegue fazer a generalização com entrada tipo AllA => (todos os parâmetros e afins fornecidos como variáveis de entrada, menos Xm, Pm)

Coisas que ficaram do dia 7:

LOSSV7 :: TODO veja se só a loss d2 é o suficiente pra representar apenas a variação do volume, que é a mais fácil. Se não, pode ter algo errado nos meus cálculos. O valor dela é até maior que a d1 e pode ser mais fácil o treinamento por d2...

-T-O-D-O-S: 
1) ORGANIZA. TEM COISA DEMAIS, MUITO LERO LERO. BOTA NO ARQUIVO DE TODOS MESMO...
2) Faz um teste tirando o volume da equação de XPS. Ele consegue dar um output OK? Claro que vai estar tecnicamente errado, mas é só pra saber se ele sai um volume variando e o xps do batelada simultaneamente
4) Bota uma variação de volume grande 3 executa num intervalo minúsculo tipo 0-1h que pareça o batelada. Preciso ver se errei no equacionamento do balanço de volume pelamor
5) Bota artigos numa pasta do zotero pra ler
3) Comece a preparar psicologicamente para ter como entrada todas as variáveis (Xo, Po, Vmax, etc). O padrão de saída do reator a gente pode manter daí não precisa de eq. saída, só da vazão de entrada mesmo.
-T-O-D-O lembre que ainda ficou tudo pelo meio de ver o pq a adimensionalização de variáveis esculhambava valores e a do tempo não dá em nada. Isso é a prioridade pra que eu possa fazer novos testes.
Comece 1) usando a loss v5 pra ver se ainda ficam resultados nada a ver quando usa nondim que não a do tempo. Preciso encontrar o erro zzzz.
Agora tenho que ver as derivadas 1ª que não fazem nenhum sentido. As 2 tão batendo ok.
Ok testei pro batch a 0.35 do tempo e deu certo nos 3. tem nada errado não zzzz.
2) fuxicando loss v7. Pela ordem, de grandeza já sei que a derivada 2 não é tão baixinha a ponto de precisar daquele super acréscimo gigantesco de 1e12. Começa por aí.


### 2023-10-18
- Posso estar errando a derivada em t=0? Como???
- Tentar loss v7J pra ver se reduz esses picos sem sentido em t próximo de zero.
- Nova auto LW ("autic") que recalcula APÓS a otimização inicial
- FEITO testa com esse novo autic
  -  Atenção: Agora a loss é feita com base no pónto mais recente, e não no primeiro. Na prática, deixou de fazer loss em i=0 e passou a fazer em i=1 para os casos do tipo "auto".
- Meu cálculos de dS/dt tá errado. Como pode dSdt ser constante e a derivada não ser 0, nem no numérico???
- Arrumei erro na derivada primeira salva do numérico (o valor calculado em si estava certo)
- Testar CR novamente com rede 200x4 e 1200 pontos train e teste. Demora uma vida...
- TODO testa com v7J usando o auto e o autic COM PRÉ-TREINO DE ICs
- TODO URGENTE veja o equacionamento e fuxique até parar de dar bode. Não é possível. Em todos dá esse pico estranho de S. Parece que o mesmo erro que tinha cometido antes, no numérico, cometi no pinn. mas simplesmente NÃO CONSIGO EXERGAR ONDE.
  - O problema antes é que eu tinha que multiplicar a conc. pelo volume antigo e dividir pelo novo. Se não fizesse isso, gerava esse acúmulo infinito. E agora tá com algo parecido. como proceder ???????? Porque o dSdt era pra ser zerado e acabou. Será que multiplicar o V ao invés de dividir resolve? Acho que não. O que fazer????????????? Multiplicar por dVdt e multiplicar por dt também? Isso existe? Tem algumn cabimento?????

### 2023-10-16

 - 1) Pegue os TODOs dos dias anteriores
 - 2) Rode o batch de maior tempo (40h) pra ver se ele também apresenta dificuldade em ser resolvido.
  - Já vi que batch muito tempo ele também desanda, então isso com certeza também tá impactando lá no cstr. Poderia rodar um tipo 0-10pa só pra constatar de fato...
  - Teve um que quase prestou em ND-Lin-UPx1-t2-F1d10 de V0-4--Vmax-5--Fin-5E0 em in_t-out_XPSV tr- 0-10pa Glorot uniform-Hammersley. Só o S saiu errado. E usou tudo (nondim, w-Squared e ic 2k)
  - SÓ REACTION  E depois batch sem volume em ts maiores pra validar t nondim.
    - São 40 resultados. Primeiro faz 8x3. Depois 8x6. Depois 30x3. E aí acaba eu acho.
    - Vai validar simultanemante: loss weights e explorar questão do tempo.
      - Então é melhor fazer todas as redes ao mesmo tempo, mas separar por nondim, pra ser menos, e aumentar LR pra enviesar menos tb.
      - O caso t6 Upx1 ficou melhor só com 10h pra predizer as 20 do que a 100%. Não acredito que só a densidade de pontos justificaria isso...
      - Por enquanto o t7 parece ser o melhor de todos, e 0-50pa MUITO melhor que 0-100pa.
  - -X- roda com outros valores de t nondim OK
  - -X- faz CRs t=> V só pra mostrar que o volume dá certo.
  - -X- roda um com Xo e Xin = 0. Isso vai acabar com a variação e ele vai precisar predizer basicamente o volume e 2 linhas retas. Não é possível.
  - -X- Olha eu poderia té continuar tentando, mas esse problema não merce tomar tanto do meu tempo. ganho mais indo terminar de fazer os ajusstes pedidos no trbalho escrito e as novas ilustrações. Não tem nem cabimento perder tanto tempo nisso. Não deu certo e pronto, acabou. Chega de sofrer com isso.

### 2023-10-15

- Parei nos outrs dias porque tava lascado de gripado
- Finalizei redes 3-9x1-6 pra batch tempo normal
-  redes 16-100x1-6 pra batch tempo normal pra LR 1e-3
  - Demorou 207.28 min = 3.45 horas
- Desisti: esse 16-100x1-6 sem nondim, loss weights trocado de "auto-e2-S" pra "A1" (sem weights, =1 pra tudo) e sem o resampler (=None).
  - Eu tenho que fazer beeem menos rede e variar mais params, pq se não a comparação é totalmente injusta.
- Então vou fazer redes minúsculas, 3-6-16 x 3-6.  Já tenho dados o suficiente pra nondim anterior, agora vou testar outras... Então faço as nondim 1 a 1 e vejo quais produzem redes boas mesmo com pouquíssimos neurônios.
  - Ameaçou prestar mas desanda fáciL: XPSV-8x3 tanh L7B LR-E-4_5 wauto-e2-S Lin-UPx1 p8-32-32 35kep lbfgs-0-1 m-
  - Continua parecendo um problema de sampling, então deveria investir mais nisso, sinceramente...
- Feito: Já seria bom eu testar uns CRs com pouca variação de volume. Aí pego só umas 3 tipos de rede, vario um pouco a LR e testo 2 nondim
  - É, ainda tá bem ruim. Mesmo com LJ e muitos pontos e resampling não deu conta.
  - Feito: implementar treinar primeiro ICs com adam 2k depois tudo como tava antes. Porque o problema de algumas é que usa as ICs erradas (loss IC alto) até o fim e só se ajeita no final, assim também não dá.
  - Fiz isso e continuou dando em NADA. Alta densidade de pontos, mais de um tipo de nondim, resampling, treino prévio de ICs. Absolutamente nada resolveu. O volume tá ok, são as outras que ele tá indo pra outro ponto de solução, como se fosse (mas sem ser exatamente) uma solução trivial.
  - Então isso dos pontos muitos já posso remover, deu em NADA.
  - mESMO REDE 80X3 TAMBÉM NÃO ROLOU
  - O valor de best loss que ele tá printando sempre é multiplicando só os pesos das ICSs, que foram os primeiros a serem usados, mas não tem nada a ver com o exibido pelo gráfico que são muito mais altos mds. Os do gráfico estão certos, é esse print usando sempre o 1º LW que tá totalmente errado.
    - Mentira, tá certo. É pq a menor loss, de fato, foi quando treinou só as ICs.
  - Bom, as redes 30x3 e 80x3 não deram conta então não é um problema meramente de mais neurônios. É uma outra coisa. E que eu não consegui resolver e vai ficar pra próxima, porque tenho que encerrar esse trabalho.
- A fazer: depois validar novamente resampler e essa estratégia de setar pesos automaticamente


-X- aí esses mesmos testes exatamente os mesmos repito pra sem nondim e pra nondim Lin-Upx1 nas redes 8X. tendo os gráficos erro e Lr e HL pras 3 já posso fazer muita conclusão.
-X- comece fazendo uma figura explicando as etapas de verificação de como funciona o PINN.
1) Screening => Validar só XPS do batelada. Ver questão da adimensionalização, loss weights e10S vs A1 e besteirol afim.
2) Volume constante e variando. Validar modelo e predição. É tão fácil que nem compensa comparar formas de adimensionalização, sinceramente. É só pra não questionarem se o volume tá errado mesmo.
3) Batch XPSV. Foi esse o que fiz por último.

- Terminar esse batch já faça o CR que varia bem pouquinho o volume, pra ver se ao menos ele funciona.


### 2023-10-11

- Obtenção de vários resultados e ver opções do matplotlib pra fazer o gráfico. Tlavez seja melhor um heatmap 2D mesmo.


### 2023-10-10

- Talvez a questão sequer seja meramente o CR, mas sim o tempo que passou de X desanda pq chega no pico de máximo. Todos aqueles artigos morrem antes disso e não tem essa inflexão de descida...  Então testar com o batch de 20h...
- Deixe rodando redes do batch pra testar o plot 3D
- NL: (2, 4, 6), (8, 10), 16, 32, 45, 64
- HL: 1-10, grupos: (1-4), (4-7), (8-10) pra maior que 16
- 4 LRs (8e-4, 1e-3, 3e-3, 5e-3,)
=> Como vai ser muito grande, faça um NL por vez e deixa o resto pronto, só vai dando comment e uncomment nos NLs
Na sequência fazer o plot 3D pra poder conversar com eles... To lerdando muito mesmo trabalhando tanto affff
- Terminei abandonando vários, demorando demaaaais.
- A loss XV3 funcionou em um dos casos mas foi MUITO epecífico, dependeu de LR até, uma faixa bem estreitinha.
- Então farei uma auto loss que roda 1 epoch e dessa 1 epoch tenta tirar a loss de cada ponto e já fazer o ajuste automático.
- FEITO: veja se os número ficam != 1, se deu certo... no json de saída
- Já tive um resultado com UPx1 que deu aquela zerada então ele sozinho não resolve não zzz

loss_ajustada = valor desejado
Ex: loss deu 1e-3 mas queria 10. O ajuste multiplicador tem que ser 1e4
Então é multiplier = scale_to/loss_firstiteration
loss_adjusted = multiplier*loss_first_iteration. Então tenho que usar a loss do 
- O auto normal favorece excessivamente quem tá errado no inicio, podendo esculhambar os outros. Tentar com raiz quadrada ao invés de puro.
- O e-2S tá MUITO melhor que o e-2, nem se compara. Já posso usar ele como base.
  Mas terminou sendo super dependente da LR. De toda forma, vou fixar o e-2S e usar ele daqui pra frente.
  - To começando a suspeitar que essa zerada é uma solução mais fácil pras redes com HL e NL muito altos, porque como tem muitos neurônios disponíveis, conseguem fazer eles "desconversarem" e cada um ter mais impacto numa região específica. Nas menores, se alguns deles desandarem pra zerar numa região pra fazer a zerada, outros podem acabar sendo obrigados a acompanhar pela dependência.

- Automatizei tudo e vou voltar a fazer de 10 em 10 pq senão ele n aguenta, fica muito lerdo. De lascar.
- Agora que to usando peso de loss adaptado, nem adianta usar a loss pra comparar, vai ter que ser na MAD mesmo ou algo equivalente.
- Fiz teste de sobrescrever json, tá funcionando bem, não estraga o arquivo.


### 2023-10-09

- Resolvi tirar as adimensionalizações parciais (ex: só as saídas ou só as entradas). Adimensionaliza logo tudo. Reduz muito os testes.
- Separar cores em arquivo em utils/colors
- Permitir plotagem de pontos extras por injeção de dependência
- Implementar plotagem dos pontos experimentais automaticamente pro batelada quando condições forem idênticas
- parece que não é nem a questão da variação de volume, mas passar um tempo grande o suficiente pra ter aquela inflexão de queda na curva. Tanto é que o batelada em tempos maiores também sofre.
- Change plots default dpi
- mudar cor pontos experimentais
- mudar pontos de treino, tem muitos de bc nem precisa. Uns 4 devem dar conta sinceramente.
- A rede 10x1 parece ter menos problema de zerar o X do que a 10x3, o perfil fica errado mas mais coerente
- Aumentar npoints definitivamente resolveu!!! 10x3 ficou pro só batch sem volume
  - Acho que já poderia tentar fazer em redes menores, tipo 4 e 8 neurons já pra produzir gráficos. E reduzir as formas de nondim já que todas deram certo com muitos pontos...
  - E tb iterar os npoints pra achar o mínimo necessário pra fazer funcionar...
- parei batelada pra fazer volume variando e ver se consigo botar pra prestar.
- era bom testar mini-batch, porque aí boto muitos pontos mas treino durante um período menor, possivelmente
  - **O mini-batch dele nem é mini-batch de fato, é um resampler**
  - E 2, não tá valendo a pena esse monte de pontos. Demora muito mais e os resultados continuam ruins...
  - Francamente, acho que trabalhar com mini-batch (que na verdade é um resampler!!!!) minúsculo pra pegar pontos novos toda iteração deve ser uma estratégia melhor do que usar pontos demais.
  - Voltei pra 300p de iteração. É mais seguro do que 32, definitavamente.
  - Upx1 parece que vai ficar melhor que o lin
  - FIXED meu Upx1 tá errado por algum motivo. Não tá criando o Upx, tá criando o lin-lin mesmo e não sei o pq.
    - Era um erro na forma de escrever upx1  tava Upx1 no lugar de UPx1 mds.

- Preciso aumentar epochs. Várias vezes corto o treino justo quando ia começar a dar certo...
- Terminei criando a 7J, que exclui aquele besteirol de signal que deve atrapalhar muito.
- talvez o que vá resolver minha vida é usar uma rede mais complexa, tipo 80x4, como padrão, 
- F1x deu Nan no UPx1. Então já posso começar pulando ele.
- Testei F1d100 no UPx1 com esperança mas também pareceu bem péssimo. Pelo visto vou ter que abandonar, foi uma ideia sem futuro.
  - Mas daí aumentei a LR e resolveu, ficou bem melhor zzzz
  - Pois vamos fazer assim: pra t2-F1d100 Lin-UPX1 varia LR e faz uma busca de HL e NL
  - Repete isso pra t2-F1d100 do LinLin
  - OK, t2x10 e F1d100 novamente ficou excelente mesmo no Lin-UPX1. Agora é bom eu testar o t2x100 só pra ver se fica ainda melhor. Isso ajuda demais... Mas novamente LR importantíssima zzz....
  - Do meio pro fim o melhor foi o t2x10F1d100. Então vou usar ele pra tudo enquanto acho uma boa...
  - NEssas settings tanto a 7B quanto a 7J deram certo, com a diferença que a 7B foi MUITO mais rápida, ficou um pouquinho pior nos gráficos normais e levemente pior no gráfico da derivada segunda.

- O volume é tão bestinha mesmo no CR que uma rede 2x1 deu conta do recado. Então ele sim pode ser visto pra ver as dificuldades de 7B, 7J e outras formas nondim. Além de ser muito mais rápido de rodar.
- A partir de hoje, tudo faz na L7B que é muito mais rápida, depois penso no caso da 7J.


-----------------------------------------

### 2023-10-08 

- !!!!!!! fui muito burro. Não podia ter tirado as strings da key porque são elas que garantem que os projetos são únicos e não vão se sobrepor. Voltando... em cases_to_try.py
- Talvez a solução seja trocar o dict por uma lista. Mas a ideia do dict era justamente proibir casos iguais usando o nome pra tirar os conflitos automaticamente...
- Trocar estrutura de dict por list. Provavelmente resolverá.
  - Decidi preservar a estrutura de dict porque ela proíbe duplicatas. Aí a key e o name são 2 coisas distintas

pinn_saver
  - Corrigido erro: MAD não era exportado
  - Corridigo erro: várias variáveis não eram exportadas no json por estarem sendo passadas com ":" e não "="
  - Mas ainda precisei dar clear no cache do python (VSCode => control shift p => clear python cache)
  - add ao json:
    - process_params
    - eq_params
    - altiok params
      - E implementar to_dict em cada respectiva classe
      - FIXED problema ref circular pinn_saver e pinn_results chamando um ao outro
        - Fiz do jeito mais fácil e burro, que foi tirar o tipo de pinn_model do pinn_saver e trabalhar "no escuro"

- cases_to_try:
  - atualizar nomenclatura e lógica de pastas, bem como ordem de preparação dos modelos
  - separar em blocos os nondim porque são muitos e ficar comentando/descomentando é trabalhoso

- Add useMathText to stylesheet
- on "plot_comparer_multiple_grid.py"
  - Apply scilimits (-1, +1) only for axis y
  - Remove previous code for settings the y lims manually (now irrelevant)
- Loss v7 (test: t => V for CR):
  - Loss = Sign of d1 => Resultados péssimos, a loss trava e não consegue mais descer (sendo que se tá lá é porque tem algo errado). talvez eu esteja usando a função sign de forma errada???
  - Testei loss = (1+sign_dif_d1)*loss_derivative_abs  e  loss = loss_derivative_abs
    - Resultados parecido, mas a loss pareceu reduzir mais rápido com o (1+sign_dif_d1)
  - Testando loss = (1) * (loss_derivative_abs + loss_minmax) e loss = (1 + sign_dif_d1) * (loss_derivative_abs + loss_minmax)
    - A deriv + loss_minmax pareceu ser PIOR que a versão sem a loss minmax. Talvez ele esteja com muito "medo" dos valores que ultrapassam os limites e isso esteja atrapalhando.
    - Novamente, os resultados que usaram o signal pareceram melhores
  - Teste rede 10x3:
    - loss = (1 + sign_dif_d1 + sign_dif_d2) * (loss_derivative_abs + loss_minmax + loss_d2) oscilou de 1 a 1.004 e loss = loss_d2 oscilou de 1 a 1.2. Sendo que os valores ficam entre 1 e 1.003. Então claramente a que tem tudo foi melhor. Agora é simplificar.
    - loss d1 solo foi pior que loss d2 solo porque fez com que tivesse aquela queda brusca no t imediatamente após 0.
    - tudo sem sign foi pior que tudo com sign..

### 2023-10-07

- Loss weight now can be iterated using dictionaries
- Implemented Upscale toNondim and fromNondim, like:
  - N_A = LB + N/N_S
  - N = N_S*(N_A - LB)
  * where N_A = nondimensional "N", N_s = "N" scaler, LB = lower bound, a parameter
- Apagar comentários
-  json solver_params => colocar nondimscaler
- upscale testar se indo e voltando dá certo
- separar nondim de input e output, agora serão 2. Talvez fique ruim só pq adimensionalizei o t tb com esses desvios estranhos
- Incluir input e output order no json de saída
- Estrutura de pasta. O que deve ir pra pasta: 
    1) tipo de reator (já tava)
    2) input string/order - output string/order => Pq agrupa por tipo de modelo
    3) range treino
    4) Init function
    5) func. distribuicao treino
    6) func. ativacao
- Agrupamento por pasta
- Remoção de conversão desnecessária de variáveis para numpy arrays
- Botar legendas do lado e não por cima nos gráficos de loss
- arredondar linha cheia da loss plot
- Reorganizar arquivo "main.py" para reaproveitar código

Loss v7
- Checar a loss. Já vi que como tava fazendo antes dNdt_2 calc e predita eram na verdade o mesmo valor, só tava aumentando o custo computacional. Tenho que fazer as contas certas e comparar corretamente.
  - Realmente, o sinal e a loss da d2 não tavam fazendo nada porque estavam subtraindo o mesmo valor e dando 0. Agora que consertei, elas pioram MUITO a loss... E olha que ainda to no batelada. Termina com loss alta e todo mundo com preguiça de mudar, em 0 ou na condição inicial.
  - 2:34 PM => parece que como montei eu estou incentivando ele a zerar a própria derivada segunda, e não a diferença entre o calculado e o predito...
  - Testar com o batch num tempo maior pra ver o que acontece e comparar as 2...
  - A lerdeza da L7 não é meramente na loss d2 existir, mas apenas quando ela é retornada na fração. Então deve ter algum valor errado, dividindo por zero ou afins que tá causando essa lerdeza. Possivelmente são erros lançados no background e que o tensorflow v1 não exibe...
  - Veja se só com o volume variando e sendo predito (in t => out V) ele presta...
    - Loss = loss_deriv1 => deu certo no geral
    - Loss = 
    - Bug no pinn saver: como tem dXdt se a saída é só V??? Resolvido. Eram dXdt e afins do NUM não do PINN. To ficando é doido.
  - FIXED: Alguns desses gráficos saem tortos por causa da faixa de variação permitida que botei. Daí quando ela é grande ele corta as bordas, sendo que a ideia era fazer o contrário (que funcionou) de tirar a escala exagerada de variações minúculas.
  - Ainda resta um problema: tá aparecendo 1e5 ao invés de x10² por exemplo. 
  ref: https://matplotlib.org/stable/gallery/ticks/scalarformatter.html#sphx-glr-gallery-ticks-scalarformatter-py
  O exemplo deles funciona perfeitamente no arquivo main, mas não funcioana dentro da função de plot em plot_comparer_multiple_grid.py
  - Deleted sciformatter
  - Ok, já descobri que é só o último plot que não pega essa formatação. Só não sei o pq.
    - Achei. Era o plt.yscale, que estava redefinindo as configurações do último ax (o de V). Não sei o pq e sinceramente não vou atrás.
- Atualizar fonte (deixar mais grossinha = mais legível)
- Mudar fonte padrão
  -  Não tava prestando, era o cache que precisei deletar. Fica em user/.matplotlib

2023-10-06 @ 6:47 PM ué rodei e pareceram OK no batelada até com a nondim to ficando é doido ????

### 2023-10-06

1) aplica tonondim e fromnondim na própria rede, como transformação de entrada e saída, pra não precisar ficar fazendo isso toda hora, só faz em um único lugar
  - Arquivo run_reactor => aplicar transformação output e input
  1.1) Output transform
    - 1.1.1) Faz a transformacao pro adimensional
    - 1.1.2) Editar no próprio ODEPreparer, não será mais necessária a etapa de desadimensionalização
      - Sem nondim deu o valor certo, tá OK como esperado (todos os scalers =1)
      - Com nondim tb mds. To começando a suspeitar que essa transform é apenas no treino, não na rede em si. Isso faz sentido??
      - Se bem que eu não tirei o to e from, então ele converte 2x e desconverte 2x, e volta pro estado inicial zzzz
      - Eu acho que vou ter que criar mais itens no from e to do nondimscaler: ddt1 e ddt2 (derivadas de 1ª e 2ª ordem em relação ao tempo, onde somente o tempo está adimensionalizado, mas as variáveis em si não.)
      - 1º vou testar usando o from no t, que é a alternativa mais fácil e ver se dá bode.
      - testar de novo sem nondim OK! Se for sem nondim do tempo. Com o tempo esculhamba tudo. O que fazer?
        - F1d10 com t1 também se esculhamba então não é bem por aí.
        - Enquanto isso t2-1 funcionou bem, então preciso rever é a adimensionalização das variáveis !!!!!!
        - Ok, o problema NÃO é na nd do tempo porque só acontece quando usa a de XPSV ou ambas. Se for só tempo (t2, t6 ou t7) não dá errado.
2) pinn_saver
  - Adição de dois novos plots: LoT e LoV das variáveis de saída
  - Novo plot: LoV condições de contorno
  - Corrigido bug que fazia todas as derivadas segundas pra plotagem usando X e não a respectiva variável de saída no index.
  - Pronto, deu tudo certo. Tá validado e funcionando pro batelada. Agora posso melhorar a lossv7 sabendo o range de valores com que to trabalhando.


2) testa com o batelada usando uma adimensionalização bem doida pra ver se sai certo.
Deu certo, ok!

1.1) Acho que eu vou ter que fazer também o gráfico de derivada segunda
  - Então seria bom já botar isso no Euler e no pinn_saver(img e json MDS)
  - Euler retornar derivadas segundas
  - Os métodos de adimensionalização devem suportar até a da derivada 2ª ????
  - Pinn saver botar no json
  - Pinn saver processar e salvar imagem
1.2) Bota uma imagem com a loss de cada saída (XPSV quando houver) ao longo do tempo. Assim eu vejo quem tá mais desviando zzzz... e uma img separada pras condições iniciais. Então são +3 imgs.
2.2) Novo teste: variando o volume, mas Cin e Co X != 0 e o resto igual a zero. Quero ver se mantém conc. constante.
- 
  - Roda uma batelada pra comparar

### 2023-10-05

*Outra coisa, me pareceu novamente que não é falta de complexidade da rede não... Algumas 8x ficaram melhores. E redes menores devem ser mais fáceis de treinar então é melhor eu realmente ir por elas e fazer só 1 ou 2 grandes pra mostrar que meramente aumentar NL ou HL não resolveria o problema. É paia mas minha conclusão pode ser sim que uma variação grande de volume não pôde ser modelada, e aí mostro esse reator simplezinho dando certo.

- Parece que desanda quando aumenta o tempo total, mesmo sem nondim de tempo. Não entendi o porquê. Mesmo que use o treinamento só num pedaço equivalente ao original. Por exemplo, a rede toda treinada de 0-15pa (~11h) em 72h pro CR que a entrada é praticamente zero fica bem ruim, mas rodar ele todo em 12h já fica bom. Sem sentido total.
  - reactor-V0-1--Vmax-5--Fin-1E-4
    - Para 0-100pa de 12h: F1d100 e F1x10 no geral ficaram ruins e t7F1d10 mostrou bons resultados
    - Para 0-15pa de 72h (~10.8h): ficou bom de novo o t7-F1d10
    - Para 0-100pa de 72h:
  - Hipótese: parte desses problemas de zerar o volume é pq é a maneira mais fácil de garantir o estado estacionário quando está chegando perto, já que zera tudo. Eu poderia fazer umas loss v7 que é igual à v5, a única diferença que d(X,P,S) não multiplica pelo volume e sim divide. E quando o volume for < 0, usar apenas um valor extremamente pequeno como 1e-10

- Refatoração do código de ODE_PREPARARER: LOSSES
- Implementação e testes da loss v7 (em progresso)
  - FIXED: apenas na loss7 dá um problema no where
    - Atualizei pra tensorflow 2 e parou. Parece que poderia ser um bug de novo e eu ia passar horas tentando contornar zzzzz.....
    - o Erro no where do V_threshold foi culpa minha, que não havia botado a última parte (O "else" do .where)
    - O tensorflow v2 tá BEM mais demorado que o 1 e emitindo muitos erros, então vou de volta pra minha terra zzz.
      - No v1 continua com o erro. então talvez o 2 lerdasse porque tá tendo um erro que ele resolvia automaticamente mas isso tinha um custo, e o v1 simplesmente joga na minha cara.
      - O erro era que eu tava retornando zero, e não zeroes_like daí dava mismatch no shape
      - No threshold do volume, acontecia algo super parecido. Eu basicamente retornava o threshold, o certo era retornar tf.ones_like(V) vezes o threshold.
        - Comparei resultados lossv7 usando ou não o absoluto como multiplicador quando o sinal da derivada predita e calculada era oposto, e pareceu representar melhor o sistema (loss maior para desvios maiores) E COISA BOA O QUE FOI PRA ZERO, FICOU COM LOSS ALTA (1E1!!!!). Isso é excelente porque torna a loss uma métrica melhor. Mas veja que não foi tanta diferença assim e dá uma lerdada no treino viu...
        - Mas aí eu só to treinando 0-15. Se treinasse os 100%, com a penalidade do erro da direção da derivada, talvez funcionasse, já que é nessa região que acontece de inverter a subida e a descida...

- Create: Loss Modules

- Voltar pro tensorflow 1 mesmo ok

- Acho que vou ter que mexer na lossv7 mais um pouco ainda... Essa ideia da soma se baseou numa coisa correta (derivadas errasdas) mas não tem sentido pq eu precisaria comparar é as derivadas segundas!! Porque o sinal das derivadas em si ele não tá errando tanto.

- Implementar derivadas segundas e testar losses antigas, que não usam. Funcionando OK lossv5 reator batelada, bons resultados.
- Agora sim testar e fazer a loss v7 no reator batelada mesmo pra saber se to fazendo direito
  - Deu certo! Batch ficou OK pra loss com a d2 e pareceu otimizar em menos epochs. Mas falta testar melhor pq foi um único teste e a 0-15pa de +/-12h né...

- Rodei pro batch a nova loss em pontos 64/64/64 e num pedacinho do batch. Até pra 6 neurônios x2 e x3 ficou bom!!! Isso já pode validar essa função loss nova viu
- Achei um artigo que escreve muito bem e muito resumido, num estilo que posso adotar pra mim. Sobre gPINN.
- O treino com essa nova loss parece ser mais estável, tem muito menos oscilações na loss.

TODO faz pro CR esses testes rápidos. Talvez me dê alguma pista.
TODO faça a lista que tá no celular. Novo gráfico, de derivadas segundas numéricas vs do PINN e botar no json.

### 2023-10-04

- Rede 20NL, lossv5:
  - t2d10-F1d10
    - O sistema que havia funcionado antes foi o de 4.5 a 5L... Por isso. É ooooutra coisa fi.
    - Perfis semelhantes à rede 30NL, mas no geral um pouco piores e loss maior (entre 1e-3 e 1e-2, antes ficava entre 1e-4 e 1e-3)
- FIX ME
  - Agora deleta normalmente todos os objetos do keras/tensorflow a cada atualização (arquivo grid_search.py).
    - ref: [Tensorflow Clear Session](https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session)
- Atualizar run_reactor e cases_totry: t range modificado e agora é só o PERCENTUAL, não o valor de t já multiplicado.
- Rodar agora sim teste relativamente grande, variando camadas, LRs, tudo no mundo, e que me dará boas respostas. SÓ NÃO PODEREI USAR TEMPO DE TREINO E PREDIÇÃO PORQUE ESTOU USANDO O COMPUTADOR ENQUANTO RODA!!!!
  - 1º: Teste com XPS de saída e t e V de entrada...
  - Realmente NÃO consegui determinar o bug dos NaNs nas condições iniciais quando alguma variável de entrada outra que o tempo é acionada. Ele alega, indiretamente, que está recebendo um tuple ou algo parecido.
- FIX: Encontrei o erro. Quando a geometrica se tornava TimeGeometry, era necessário usar uma classe diferente para DATA, bem como condições de contorno diferentes. Arquivos atualizados: run_reactor, cases_to_try. Também modifiquei parte a lógica de determinar o % a ser explorado no treino em relação ao máximo.
- Novo reator padrão: reactor-V0-1--Vmax-5--Fin-25E-2
  - Acho que a rede 10x é o suficiente. Tenho que explorar é estratégias. Então faça nela que vc consegue testar mais coisas.
  - selu não ficou muito legal, nem swish mas não rodei demais. Terminei decidindo rodar só em 2 LR (1e-3 e 1e-4) pra facilitar minha vida e conseguir fazer mais testes.
  - Fiz testes com 10k adam tb pareceu pouco. Vou apagar e fazer mais com mais pontos.
  - Novos testes zzz.. Se ficarem ruins, nos outros aumentar pra 32 NL porque já tive bons resultados com 32x3 e 32x5 no passado.
  - Ainda parecem bem ruins, swish e tanh. Usei nondim tempo mas nos testes antigos todos os nondim tempo tinham ficado ruins. Nos novos tinham uns bons pq não fiz para o reator CR, só isso e fim. Então é melhor eu tentar sem mesmo...
  - Fiz mais e continuam péssimos...


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