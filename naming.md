
# Naming convention

- Loss function
  - 1 é a loss tradicional, #2 retorna o própria X/P/S/V caso seja menor que zero.
  - 3 é a que faz dMols/dt e não dConc/dt, pro volume ir multiplicando... E também faz com que EFETIVAMENTE X NaN seja bloqueado
  - A loss v4 é a que faz com que EFETIVAMENTE retorne o próprio valor da coisa (XPSV) se for < 0 ou maior que o limite (Xm, Pm, So. Volume fica solto.). E a loss v4 também é absoluta
  - 5 é a que inclui a soma de XPSV qwuando ultrapassem o limite mas ajusta fazer ^3 para valores abaixo de 0
  - 6 está em desenvolvimento mas é uma simplificação da #5 pra ver se a #5 se justifica.
  - 7 é a que não faz o volume multiplicando, mas divindo para evitar ele querer zerar o volume pra chegar no estado estacionário.

Loss Teste/Validation => LoV
Loss Train => LoT

Learning Rate => LR


- Reatores:
  - R => Reação sem nada
  - B => Batelada
  - CSTR => É de fato um CSTR, já inicia no volume máximo e só vamos ver as concentrações variando.
  - CR1 => CR que possui Vo = 1L
  - CR2 => CR que possui Vo = 0.1L
  - $t_TR$ (t subscript TR) em gráficos => o tempo discretizado usado para treinos
  - $t_SIM$ (t subscript SIM) em gráficos => o tempo discretizado usado para a simulação em si (predição após treino)
