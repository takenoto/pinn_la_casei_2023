# Thinking

## Gráficos chave

1) Loss vs epochs vs LR para alguns tipos de redes selecionados
2) Comparar diferentes nondim de alguma forma. 3D com todas e loss talvez seja o mais prático mas pode ficar difícil de enxergar. Preciso testar.

## Experiments

### parte 1 - reaction

1) Faz a nondim de 10x3 só com o tempo, sem XPS
2) Daí acho que eu já posso escolher uns bons e fazer LR e os gráficos 3D
3) LR vs Loss vs ts para: 
  3.1) sem nondim
  3.2) nondim em 0 a 1
  3.3) nondim em 0 a 10

- TODO 
  - Agora faz pra rede 10x3 e todos os lrs, as nondim:
    - ("Lin", "t6", "F1d10"),
    - ("None", "t1", "1"), 
    - ("Lin", "t7", "1"),

### parte 2 - reactor volume

1) Faz testes nos valores de reaction que já sei que vai ficar bom. Preciso validar o volume para cstr e CRs 1 e 2... Pra ver se de fato tá tudo ok viu...

- Reaction
  - Find an OK LR/Adam/NL/HL com teste relativamente aberto (não faz muito fininho não)
  - Depois acho um bonzinho e faço variações ao redor dele. Faz mais sentido.
  - 1º roda só 1 adam. Daí Quando terminar é que roda o seguinte.
  - 0-60% dá certo rapidinho, é impressionante. 0-90 já desanda. Parece que esse plateau do final influencia muito em tornar o sistema uma reta mds.
  - Adam só 100 iterações pra 30x3 seguida de 1 lbfgs já ficou ótimo a 0-90pa. NONDIM: ("Lin", "t2", "1"), ou seja, tnondim simples entre 0 e 1 (tmax).
  - Diferença de Hammersley pra uniform é absurda. Um dá totalmente certo e o outro totalmente errado!!!!!!!!