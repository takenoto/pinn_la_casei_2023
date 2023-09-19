# Thinking

## Experiments

### parte 1 - reaction

1) Faz a nondim de 10x3 só com o tempo, sem XPS
2) Daí acho que eu já posso escolher uns bons e fazer LR e os gráficos 3D

### parte 2 - reactor volume

1) Faz testes nos valores de reaction que já sei que vai ficar bom. Preciso validar o volume para cstr e CRs 1 e 2... Pra ver se de fato tá tudo ok viu...

- Reaction
  - Find an OK LR/Adam/NL/HL com teste relativamente aberto (não faz muito fininho não)
  - Depois acho um bonzinho e faço variações ao redor dele. Faz mais sentido.
  - 1º roda só 1 adam. Daí Quando terminar é que roda o seguinte.
  - 0-60% dá certo rapidinho, é impressionante. 0-90 já desanda. Parece que esse plateau do final influencia muito em tornar o sistema uma reta mds.
  - Adam só 100 iterações pra 30x3 seguida de 1 lbfgs já ficou ótimo a 0-90pa. NONDIM: ("Lin", "t2", "1"), ou seja, tnondim simples entre 0 e 1 (tmax).
  - Diferença de Hammersley pra uniform é absurda. Um dá totalmente certo e o outro totalmente errado!!!!!!!!