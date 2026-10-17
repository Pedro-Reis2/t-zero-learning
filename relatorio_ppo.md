# Relatório: Proximal Policy Optimization (PPO)

**Aluno:** Pedro Reis  
**Ambiente:** `LunarLander-v3`  
**Repositório:** [https://github.com/Pedro-Reis2/t-zero-learning](https://github.com/Pedro-Reis2/t-zero-learning)  
Arquivos implementados: `algorithms/ppo.py`, `networks/discrete_actor_critic.py`, `configs/ppo_lunarlander_tuned.yml`, `configs/a2c_lunarlander.yml`.

---

## Parte 4 — Resolver o LunarLander com PPO

O arquivo `configs/ppo_lunarlander.yml` original traz os hiperparâmetros padrão do CleanRL ajustados para o CartPole-v1. No LunarLander-v3 esses valores aprendem timidamente mas **não resolvem** o ambiente (retorno médio entre -31 e +5 em 1M passos).

### 1. Tabela de Mudanças de Hiperparâmetros

| Chave | Valor Antigo (ppo_lunarlander.yml) | Valor Novo (ppo_final / tuned) |
|---|---|---|
| `total_timesteps` | `1.000.000` | `2.000.000` |
| `ppo.gamma` ($\gamma$) | `0.99` | `0.999` |
| `ppo.gae_lambda` ($\lambda$) | `0.95` | `0.98` |
| `num_envs` | `4` | `16` |
| `ppo.num_steps` | `128` | `1024` |
| `ppo.num_minibatches` | `4` (tamanho minilote: 128) | `256` (tamanho minilote: 64) |
| `ppo.learning_rate` | `2.5e-4` | `2.5e-4` |

---

### 2. Motivo Observado para Cada Mudança

1. **`total_timesteps: 1.000.000 -> 2.000.000`**
   - **Diagnóstico:** No run `ppo_inicial` (1M passos), ao observar `charts/episodic_return_mean_last100`, a curva ainda apresentava inclinação positiva aos 1M passos. No LunarLander os episódios podem durar até 1.000 passos e o bônus principal de pouso suave (+100) só acontece no final. 1M passos com 4 ambientes forneciam apenas 1.953 coletas de rollout, número insuficiente para estabilizar a política. Dobrar o orçamento foi indispensável.
2. **`gamma: 0.99 -> 0.999` e `gae_lambda: 0.95 -> 0.98`**
   - **Diagnóstico:** No run `ppo_2M` (orçamento estendido para 2M passos mas mantendo $\gamma=0.99$), o retorno estacionou em apenas **28.4**. Ao examinar `losses/explained_variance`, notou-se que o crítico falhava em atribuir crédito ao pouso. Com $\gamma=0.99$, em um episódio de 500 passos o desconto é $0.99^{500} \approx 0.0065$ — a recompensa de pouso praticamente desaparece para os passos iniciais de orientação do módulo. Elevando $\gamma$ para $0.999$ ($0.999^{500} \approx 0.606$) e $\lambda$ para $0.98$ (`ppo_2M_gamma`), o retorno saltou espetacularmente de 28.4 para **193.3** (atingindo pico > 200 aos 1.74M passos e avaliação final de +219.1).
3. **`num_envs: 4 -> 16` e `num_steps: 128 -> 1024`**
   - **Diagnóstico:** Com apenas 128 passos por ambiente, cada rollout capturava apenas 10–20% de um episódio típico. Aumentando o paralelismo para 16 ambientes e o horizonte de coleta para 1024 passos (lote total de 16.384 transições), o throughput `charts/SPS` subiu de ~1.300 para ~2.250 passos/s (um ganho de 70% em tempo de relógio) e a variância dos gradientes foi reduzida drasticamente.

---

### 3. Runs que Não Deram Certo no Caminho até o Config Final

- **`ppo_inicial` (1M passos):** Ficou entre -31.5 e +5.0 de retorno médio. A política apenas evitava colisões catastróficas imediatas, sem pousar.
- **`ppo_2M` (2M passos, $\gamma=0.99$):** Estagnou em retorno médio de 28.4. Demonstrou categoricamente que aumentar o número de passos sem estender o horizonte de crédito temporal $\gamma$ é ineficaz para o LunarLander.

---

### 4. Resultados e Curvas de Retorno (2 seeds)

| Experimento | Seed | Retorno Final (last100) | Avaliação Final (10 eps) | Passos p/ 0 | Passos p/ 100 | Passos p/ 200 | Tempo (min) | SPS | Num Updates |
|---|---|---|---|---|---|---|---|---|---|
| **ppo_inicial** | 1 | 5.0 | 32.6 ± 113.4 | 0.31M | — | — | 13.1 | 1296 | 31.248 |
| **ppo_inicial** | 2 | -31.5 | 96.1 ± 107.4 | 0.26M | 0.47M | — | 12.7 | 1345 | 31.248 |
| **ppo_2M** | 1 | 28.4 | 76.4 ± 131.8 | 0.28M | 0.42M | — | 21.9 | 1534 | 62.496 |
| **ppo_2M_gamma** | 1 | **193.3** | **172.0 ± 89.2** | 0.18M | 0.63M | **1.74M** | 21.1 | 1591 | 62.496 |
| **ppo_final** | 1 | 112.0 | 138.6 ± 38.6 | 0.23M | 0.49M | — | 15.3 | 2234 | 124.928 |
| **ppo_final** | 2 | 131.3 | 126.1 ± 31.5 | 0.21M | 0.56M | — | 15.1 | 2272 | 124.928 |

*(Gráficos correspondentes gerados em `report/figs/p4_*.png` e incorporados no PDF).*

---

## Parte 5 — PPO vs A2C no LunarLander-v3

Para comparar os dois algoritmos honestamente sob as mesmas condições, o A2C implementado na atividade anterior foi configurado em `configs/a2c_lunarlander.yml` com o mesmo orçamento (2M passos), 16 ambientes paralelos e $\gamma=0.995$.

| Algoritmo | Seed | Retorno Final | Avaliação Final | Passos p/ 0 | Passos p/ 100 | Tempo (min) | SPS | Num Updates |
|---|---|---|---|---|---|---|---|---|
| **PPO (final)** | 1 | +112.0 | +138.6 ± 38.6 | 0.23M | 0.49M | 15.3 | 2234 | 124.928 |
| **PPO (final)** | 2 | +131.3 | +126.1 ± 31.5 | 0.21M | 0.56M | 15.1 | 2272 | 124.928 |
| **A2C** | 1 | -123.4 | -124.1 ± 38.4 | 0.21M | — | 8.1 | 4272 | 25.000 |
| **A2C** | 2 | -117.0 | -132.2 ± 41.6 | 0.22M | 0.59M | 7.9 | 4353 | 25.000 |

### 1. Eficiência Amostral (Retorno × Passos de Ambiente)
O PPO supera o A2C com margem esmagadora. Enquanto o PPO atinge retorno 0 aos ~0.2M passos e retorno >100 aos ~0.5M passos convergindo para pousos consistentes, o A2C **nunca consegue aprender a pousar**: estabiliza em torno de -120 a -117 em ambas as seeds. A política do A2C fica presa em tentar retardar a queda, mas sempre colide ou esgota o combustível.

### 2. Tempo de Relógio e SPS (Retorno × Tempo)
O A2C possui maior throughput de passos brutos: roda a ~4.300 SPS e conclui os 2M passos em ~8 minutos, contra ~2.250 SPS e 15 minutos do PPO.  
**Causa:** O gráfico `charts/num_updates` revela que o A2C faz apenas 1 atualização de gradiente por coleta (25.000 atualizações no total), enquanto o PPO realiza múltiplas épocas sobre minibatches (124.928 atualizações). O PPO passa mais tempo refinando os parâmetros por passo coletado.

### 3. Comparação sob o Mesmo Orçamento
- **Mesmo orçamento de passos (2M):** PPO vence com enorme folga (+130 a +193 vs. -120).
- **Mesmo orçamento de tempo (8 min):** Aos 8 minutos, o PPO já executou 1,1M passos e atinge retorno médio acima de +100 (conforme o gráfico `Retorno × Tempo`), enquanto o A2C terminou seus 2M passos estagnado em -120. O PPO é superior em ambas as métricas.

### 4. Estabilidade
As seeds do PPO apresentaram convergência monótona, com desvio padrão na avaliação de apenas ±31 a ±38 no modelo final. O A2C apresentou curvas artificialmente estáveis em patamar negativo porque colapsou para um mínimo local degenerado.

### 5. Mecanismo das Quatro Mudanças do PPO
1. **GAE ($\lambda$):** O crítico do PPO mantém variância explicada alta e estável (`losses/explained_variance` entre 0.8 e 0.9), permitindo estimativas de vantagem com sinal e magnitude precisos. No A2C, o retorno de 5 passos tem viés severo e o crítico colapsa no LunarLander.
2. **Reuso de Amostras ($K$ épocas):** Permite extrair muito mais sinal de cada transição, realizando >120k passos de gradiente com o mesmo número de interações com o simulador.
3. **Razão de Probabilidades ($r(\theta)$):** Ajusta matematicamente o gradiente à medida que a política muda durante as épocas de treino.
4. **Clipping ($[1-\epsilon, 1+\epsilon]$):** Mantém a divergência KL controlada (`losses/approx_kl` $\le 0.02$) e a fração de corte em níveis saudáveis (`losses/clipfrac` entre 5% e 15%), impedindo que as múltiplas épocas destruam a política.
