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

### 2023-09-09

- Loss v5 foi modificada, e sem a normalização automática. Resultados absurdamente melhores, praticamente todas as redes ficaram boas e com baixa variação, até mesmo do volume. Essa versão já pode ser a loss final porque acabou de ser validade pro reator batelada.
- Loss v5 validada com batelada.
- CR => Novos modelos definidos no código e no arquivo naming.
- Testei distribuições e salvei (Hammersley e LHS) mas não parece ser aí o problema
- Mesmo a versão CSTR puro (já inicia no volume máximo) fica bem péssimo. O volume até prediz ok, o resto desanda. Continua me parecendo que esse pico da função é super responsável por isso... O batch já validou o equacionamento da reação... Então isso tá certo, já foi confirmado. Ficou super bom mesmo nessa lossv5.
- Cheguei a testar 4000 pontos teste e domain + mini-batch de 20, continuou bem ruim.



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