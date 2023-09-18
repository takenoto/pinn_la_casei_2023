# Thinking

## Experiments

- Reaction
  - Find an OK LR/Adam/NL/HL com teste relativamente aberto (não faz muito fininho não)
  - Depois acho um bonzinho e faço variações ao redor dele. Faz mais sentido.
  - 1º roda só 1 adam. Daí Quando terminar é que roda o seguinte.
  - 0-60% dá certo rapidinho, é impressionante. 0-90 já desanda. Parece que esse plateau do final influencia muito em tornar o sistema uma reta mds.
  - Adam só 100 iterações pra 30x3 seguida de 1 lbfgs já ficou ótimo a 0-90pa. NONDIM: ("Lin", "t2", "1"), ou seja, tnondim simples entre 0 e 1 (tmax).
  - Diferença de Hammersley pra uniform é absurda. Um dá totalmente certo e o outro totalmente errado!!!!!!!!