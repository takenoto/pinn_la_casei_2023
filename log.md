# Log

## Testes a se fazer
- Batch
  - scaler 1/10 pra tudo, incluindo o tempo com t_nondim, mas pra V scaler 1. Ajuda a ficar melhor????
- CSTR
  - As melhores redes do batch...
  - Loss v4 com os 3 nondims tanh 60k adam
  - Tentar scaler 1/10 pra todos mas 1 pro volume. 10 pro tempo vai cancelar então tb daria certo...
  - Repetir mas com swish


## Anotações
- Loss v4
  - Batch
    - Já fiz vários nd com t e sem t e linear e vi que muitos ficam ok.; A loss v4 parece estimular o volume a ficar caindo e não oscilando.
    - None e Desvio validados! Desvio não ficou tão bom mas pode ter a ver com a minha escolha de scaler pras variáveis.
  - Adimensionalização + scaling pra deixar todo mundo entre 0 e 10 parece a melhor opção até agora. Excluindo sempre o tempo da adimensionalização, claro... Será que fica bom entre 0 e 10 justamente por ser o mesmo range de variação de t_final? Que vai até 10.2...
  - É, mas também foi o único em que eu botei pra ir até 55k otimização. Isso deve contar mais que todo o resto zzzzzzzz.... Os outros eram só 35k eu acho.


## by date

### 2023-08-22
- Encontrei um erro sério
  - Ao pegar o reator, sempre convertia como se tivesse adimensionalizado. Mas acontece que o método numérico nunca tava sendo adimensionalizado. I.E. os valores nunca são os nondim, mas os valores absolutos (NO NUMÉRICO). Isso significa que tudo de lá deve ser convertido para nondim.
- Adicionei o salvamento dos valores calculados numericamente, predição pinn e predição pinn ainda nondim, sem converter. Pro caso de precisar no futuro. Não é possível, agora tá fácil de mexer. Pelo amor...

### 2023-08-23
- Novo teste loss v4 + nondim batch: 6. Deu certo. O único problema é o volume caindo ao longo do tempo.