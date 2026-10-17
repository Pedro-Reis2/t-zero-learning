"""Generate the SAC assignment PDF report (relatorio_sac.pdf, <= 4 pages) using ReportLab.

Embeds exact tables, run IDs, and figures from report_sac/figs/.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

PDF_PATH = "relatorio_sac.pdf"
FIGS = Path("report_sac/figs")

doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=letter,
    leftMargin=32,
    rightMargin=32,
    topMargin=26,
    bottomMargin=26,
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontSize=15,
    leading=18,
    textColor=colors.HexColor("#1a365d"),
    spaceAfter=2,
)
subtitle_style = ParagraphStyle(
    "DocSubtitle",
    parent=styles["Normal"],
    fontSize=8.2,
    leading=11,
    textColor=colors.HexColor("#2d3748"),
    spaceAfter=4,
)
h1_style = ParagraphStyle(
    "H1",
    parent=styles["Heading2"],
    fontSize=11.5,
    leading=14,
    textColor=colors.HexColor("#2b6cb0"),
    spaceBefore=4,
    spaceAfter=2,
    keepWithNext=True,
)
h2_style = ParagraphStyle(
    "H2",
    parent=styles["Heading3"],
    fontSize=9.6,
    leading=12,
    textColor=colors.HexColor("#1a202c"),
    spaceBefore=4,
    spaceAfter=2,
    keepWithNext=True,
)
body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=8.1,
    leading=10.8,
    textColor=colors.HexColor("#2d3748"),
    spaceAfter=3,
)
table_cell = ParagraphStyle(
    "TableCell",
    parent=styles["Normal"],
    fontSize=7.2,
    leading=9.0,
    textColor=colors.HexColor("#2d3748"),
)
table_header = ParagraphStyle(
    "TableHeader",
    parent=styles["Normal"],
    fontSize=7.3,
    leading=9.1,
    fontName="Helvetica-Bold",
    textColor=colors.whitesmoke,
)

story = []

# ==================== PÁGINA 1: CABEÇALHO, SANITY CHECK E PARTE 3 ====================
story.append(Paragraph("Relatório Experimental: Soft Actor-Critic (SAC)", title_style))
story.append(Paragraph(
    "<b>Aluno:</b> Pedro Reis &nbsp;|&nbsp; "
    "<b>Ambiente:</b> <code>Hopper-v5</code> (100.000 passos)<br/>"
    "<b>Fork GitHub (<code>algorithms/sac.py</code> e configs):</b> "
    "<a href='https://github.com/Pedro-Reis2/t-zero-learning' color='#2b6cb0'><u>https://github.com/Pedro-Reis2/t-zero-learning</u></a><br/>"
    "<b>Projeto Weights &amp; Biases (todos os runs públicos):</b> "
    "<a href='https://wandb.ai/pedroreis2-federal-university-of-goi-s/sac-assignment' color='#2b6cb0'><u>https://wandb.ai/pedroreis2-federal-university-of-goi-s/sac-assignment</u></a>",
    subtitle_style,
))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e0"), spaceAfter=4))

story.append(Paragraph(
    "<b>Implementação e Verificação (Partes 1 e 2):</b> As funções <code>squashed_gaussian_sample</code> "
    "(<code>algorithms/sac.py</code>, linhas 145–155, amostragem reparametrizada via <code>rsample()</code> e correção "
    "jacobiana do <code>tanh</code>) e <code>compute_soft_q_target</code> (linhas 179–180, alvo pessimista "
    "<i>r + &gamma;(1 - d)(min(Q'<sub>1</sub>, Q'<sub>2</sub>) - &alpha; log &pi;(a'|s'))</i>) passaram nos 16 testes de "
    "<code>tests/test_sac.py</code>. No sanity check <code>sac_pendulum</code> (run <code>zexmnmhh</code>, 20k passos), "
    "o agente obteve <code>eval/mean_return = -165.5 &plusmn; 51.6</code> e <code>eval/mean_return_sampled = -150.3 &plusmn; 72.5</code> (&gt; -200).",
    body_style,
))

story.append(Paragraph("Parte 3 — Eficiência Amostral no Hopper-v5 (100 mil passos, 3 seeds por config)", h1_style))
story.append(Paragraph(
    "Partindo de <code>configs/sac_hopper.yml</code> (<code>sac_hopper</code>), avaliamos 4 candidatos com 3 seeds "
    "(<code>seed=1, 2, 3</code>) e orçamento exato de 100.000 passos. A tabela abaixo resume cada mudança, o motivo "
    "observado citando o run específico no W&amp;B, as métricas finais como <b>média (mín–máx)</b> e o veredito:",
    body_style,
))

p3_rows = [
    [
        Paragraph("Configuração<br/>(Runs no W&amp;B)", table_header),
        Paragraph("Mudança<br/>(antigo &rarr; novo)", table_header),
        Paragraph("Motivo observado (citando o run onde foi visto)", table_header),
        Paragraph("eval/mean_return<br/>média (mín–máx)", table_header),
        Paragraph("r_last100 final<br/>média (mín–máx)", table_header),
        Paragraph("Veredito", table_header),
    ],
    [
        Paragraph("<b>sac_hopper</b> (Inicial/Final)<br/><code>ss0o8ouj</code>, <code>qbrrtmhs</code>, <code>5rz11r8a</code>", table_cell),
        Paragraph("Padrão CleanRL<br/>(<code>ls=5000, tau=0.005, pf=2, utd=1</code>)", table_cell),
        Paragraph("Config inicial de referência; alta variância entre seeds na avaliação final (<code>20.2</code> no s3 vs <code>2462.5</code> no s2).", table_cell),
        Paragraph("<b>1546.8</b><br/>(20.2–2462.5)", table_cell),
        Paragraph("<b>1163.7</b><br/>(862.2–1334.8)", table_cell),
        Paragraph("<b>Referência<br/>(Config Final)</b>", table_cell),
    ],
    [
        Paragraph("<b>sac_ls1000</b><br/><code>vfc8ax8e</code>, <code>f2vi8mh6</code>, <code>jy4eaw0u</code>", table_cell),
        Paragraph("<code>learning_starts</code>:<br/>5000 &rarr; 1000", table_cell),
        Paragraph("No run <code>ss0o8ouj</code> (<code>sac_hopper</code> s1), os primeiros 5.000 passos (5% do orçamento) foram gastos em coleta aleatória com <code>r_last100</code> estagnado em ~18, motivando iniciar o treino em 1.000 passos.", table_cell),
        Paragraph("<b>947.9</b><br/>(733.9–1374.3)", table_cell),
        Paragraph("<b>775.5</b><br/>(725.3–805.1)", table_cell),
        Paragraph("<font color='#c53030'><b>Piora</b></font><br/>(<code>r_last100</code> &lt; 862.2)", table_cell),
    ],
    [
        Paragraph("<b>sac_tau001</b><br/><code>11gntid3</code>, <code>1z94x8bs</code>, <code>tux9q718</code>", table_cell),
        Paragraph("<code>sac.tau</code>:<br/>0.005 &rarr; 0.01", table_cell),
        Paragraph("No run <code>qbrrtmhs</code> (<code>sac_hopper</code> s2), <code>losses/qf1_values</code> levou ~60k passos para subir a ~170 enquanto o retorno estagnou em ~330 entre 30k–60k passos, motivando dobrar a taxa Polyak dos alvos.", table_cell),
        Paragraph("<b>862.1</b><br/>(709.2–944.4)", table_cell),
        Paragraph("<b>818.7</b><br/>(536.1–1110.1)", table_cell),
        Paragraph("<font color='#b7791f'><b>Indistinguível / Piora</b></font>", table_cell),
    ],
    [
        Paragraph("<b>sac_pf1</b><br/><code>yga0jegn</code>, <code>zyiwns5w</code>, <code>ilp879gd</code>", table_cell),
        Paragraph("<code>policy_frequency</code>:<br/>2 &rarr; 1", table_cell),
        Paragraph("No run <code>ss0o8ouj</code> (<code>sac_hopper</code> s1), atualizar o ator só a cada 2 passos do crítico atrasou a descida de <code>losses/actor_loss</code> até ~50k passos, motivando atualizar o ator em todo passo.", table_cell),
        Paragraph("<b>2124.9</b><br/>(985.3–2988.3)", table_cell),
        Paragraph("<b>1005.7</b><br/>(913.9–1157.1)", table_cell),
        Paragraph("<font color='#2b6cb0'><b>Indistinguível do ruído</b></font>", table_cell),
    ],
    [
        Paragraph("<b>sac_utd2</b><br/><code>aebnfakf</code>, <code>f6l8uj5v</code>, <code>dglldzup</code>", table_cell),
        Paragraph("<code>updates_per_step</code>:<br/>1 &rarr; 2", table_cell),
        Paragraph("Nos runs <code>ss0o8ouj</code> e <code>qbrrtmhs</code>, <code>r_last100</code> ainda crescia quase linearmente aos 100k passos sem atingir platô com 94.900 gradientes, motivando dobrar para 2 atualizações/passo (189.800 gradientes).", table_cell),
        Paragraph("<b>1200.2</b><br/>(1166.3–1257.5)", table_cell),
        Paragraph("<b>1292.2</b><br/>(981.6–1528.1)", table_cell),
        Paragraph("<font color='#2b6cb0'><b>Indistinguível do ruído</b></font>", table_cell),
    ],
]

t_p3 = Table(p3_rows, colWidths=[86, 80, 172, 74, 68, 68])
t_p3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7fafc"), colors.white]),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
]))
story.append(t_p3)
story.append(Spacer(1, 4))

story.append(Paragraph(
    "<b>Análise e Escolha do Config Final:</b> Iniciar o treino em 1.000 passos (<code>sac_ls1000</code>) causou sobreajuste "
    "precoce nos poucos episódios iniciais do buffer, deixando todo o intervalo de <code>r_last100</code> (725.3–805.1) "
    "abaixo do mínimo do config inicial (862.2). Aumentar <code>tau=0.01</code> (<code>sac_tau001</code>) reduziu as médias "
    "e elevou o viés <code>charts/q_minus_return</code> médio de -16.9 para +6.5. O candidato <code>sac_pf1</code> obteve "
    "a maior média em <code>eval/mean_return</code> (2124.9 vs 1546.8), mas com ampla sobreposição de faixa e média menor em "
    "<code>r_last100</code> (1005.7, faixa 913.9–1157.1 totalmente contida em 862.2–1334.8). Inversamente, <code>sac_utd2</code> "
    "obteve a maior média em <code>r_last100</code> (1292.2), mas média menor em <code>eval/mean_return</code> (1200.2) e "
    "faixas sobrepostas, custando o dobro do tempo (83,3 min vs 42,1 min). Como nenhum candidato superou o ruído entre seeds, "
    "<b>o config final é o próprio config inicial (<code>sac_hopper</code>)</b>. A Figura abaixo mostra o config inicial/final "
    "e os dois candidatos que chegaram mais perto (<code>sac_pf1</code> e <code>sac_utd2</code>, 3 seeds cada):",
    body_style,
))

story.append(Image(str(FIGS / "p3_return_inicial_vs_final.png"), width=530, height=166))

story.append(PageBreak())

# ==================== PÁGINA 2: PARTE 4 (ABLAÇÕES 1 E 2) ====================
story.append(Paragraph("Parte 4 — O que cada peça faz (Ablações em Hopper-v5, 2 seeds cada)", h1_style))
story.append(Paragraph(
    "Partindo do config final da Parte 3 (<code>sac_hopper</code>, seeds 1 e 2: <code>ss0o8ouj</code> e <code>qbrrtmhs</code>, "
    "média <code>eval/mean_return = 2310.2</code> e <code>r_last100 = 1078.2</code>), executamos as duas ablações com 2 seeds cada.",
    body_style,
))

story.append(Paragraph("1. Ablação 1 — Sem ajuste automático da temperatura (sac.autotune=false, sac.alpha=0.08)", h2_style))
story.append(Paragraph(
    "<b>Escolha de <code>sac.alpha=0.08</code>:</b> Nos 3 seeds de <code>sac_hopper</code> (<code>ss0o8ouj</code>, "
    "<code>qbrrtmhs</code>, <code>5rz11r8a</code>), <code>losses/alpha</code> parte de exp(0) = 1.0, cai para ~0.10 aos 15k "
    "passos e oscila entre 0.06 e 0.11 no restante do treino, terminando em 0.0728, 0.0805 e 0.0750 (média 0.0761). "
    "Escolhemos portanto <code>sac.alpha=0.08</code> como temperatura fixa representativa do regime estacionário.<br/>"
    "<b>Previsão (antes do experimento):</b> Esperávamos que, mesmo fixando &alpha; = 0.08 no valor assintótico, a falta de "
    "&alpha; alto nos primeiros 15k passos e a ausência de realimentação fariam <code>losses/entropy</code> desviar de "
    "<code>charts/target_entropy = -3.0</code>, prejudicando a exploração e reduzindo o retorno.<br/>"
    "<b>O que aconteceu:</b> A previsão <b>se confirmou</b>. Enquanto <code>sac_hopper</code> (azul) ajusta <code>losses/alpha</code> "
    "(painel a) e mantém <code>losses/entropy</code> cravado em <code>target_entropy = -3.0</code> dos 15k aos 100k passos (painel b), "
    "nos runs sem autotune (<code>to1o22an</code> e <code>xztzgnmu</code>, vermelho) a entropia fica sem controle: no seed 1 "
    "(<code>to1o22an</code>) despenca até -4.3 entre 30k–60k passos (política prematuramente determinística, "
    "<code>eval/mean_return = 317.0</code>), e no seed 2 (<code>xztzgnmu</code>) fica acima do alvo em ~-2.1 "
    "(<code>eval/mean_return = 1087.7</code>). A média de <code>eval/mean_return</code> caiu de <b>2310.2 para 702.4</b> "
    "(e <code>r_last100</code> caiu de 1078.2 para 867.0).<br/>"
    "<b>Explicação no código (<code>algorithms/sac.py</code>):</b> Na <b>linha 341</b> "
    "(<code>self.alpha = self.log_alpha.exp().item() if cfg.autotune else float(cfg.alpha)</code>), <code>autotune=False</code> "
    "congela <code>self.alpha = 0.08</code>, e o bloco <code>if cfg.autotune:</code> nas <b>linhas 489–497</b> de "
    "<code>_update()</code> deixa de otimizar <code>compute_alpha_loss</code> (<b>linhas 198–206</b>: "
    "<code>(-log_alpha.exp() * (log_pi.detach() + target_entropy)).mean()</code>), desativando a realimentação que aumenta "
    "&alpha; quando <i>-E[log &pi;] &lt; -3.0</i> e reduz &alpha; quando <i>-E[log &pi;] &gt; -3.0</i>.",
    body_style,
))

story.append(Image(str(FIGS / "p4_ablation_autotune.png"), width=540, height=143))
story.append(Spacer(1, 6))

story.append(Paragraph("2. Ablação 2 — Um crítico só (sac.clipped_double_q=false)", h2_style))
story.append(Paragraph(
    "<b>Previsão (antes do experimento):</b> Esperávamos que usar apenas <i>Q<sub>1</sub></i> no lugar de "
    "<i>min(Q<sub>1</sub>, Q<sub>2</sub>)</i> gerasse forte viés de superestimação (<code>charts/q_minus_return &gt; 0</code> "
    "e <code>losses/qf1_values</code> inflados), degradando o retorno porque o ator explora picos espúrios de <i>Q<sub>1</sub></i>.<br/>"
    "<b>O que aconteceu:</b> A previsão <b>se confirmou de forma contundente</b>. Na Figura abaixo, enquanto no "
    "<code>sac_hopper</code> (azul) o viés <code>charts/q_minus_return</code> permanece levemente negativo (média -27.6 em "
    "<code>ss0o8ouj</code> e -6.9 em <code>qbrrtmhs</code>, painel a) e <code>losses/qf1_values</code> estabiliza em ~216 "
    "(painel b), nos runs com um crítico só (<code>w9vi5gx6</code> e <code>ubf3qpuq</code>, vermelho) "
    "<code>charts/q_minus_return</code> dispara para <b>+300 a +620</b> (média +171.1 e final +251.3 em <code>w9vi5gx6</code>; "
    "média +291.2 e final +442.2 em <code>ubf3qpuq</code>) e <code>losses/qf1_values</code> mais que dobra (409.0 e 491.4). "
    "No seed 2 (<code>ubf3qpuq</code>, painel c), quando a superestimação passa de +300 (~35k passos), o retorno colapsa de "
    "450 para 10 e termina com <code>eval/mean_return = 48.6</code> (média <b>745.7</b> nos 2 seeds vs <b>2310.2</b> no baseline).<br/>"
    "<b>Explicação no código (<code>algorithms/sac.py</code>):</b> Na <b>linha 367</b> "
    "(<code>pick_q2 = (lambda q1, q2: q2) if cfg.clipped_double_q else (lambda q1, q2: q1)</code>), "
    "<code>clipped_double_q=False</code> faz <code>pick_q2</code> devolver <code>q1</code> no lugar de <code>q2</code>. "
    "Com isso, tanto o alvo na <b>linha 455</b> (<code>compute_soft_q_target</code>, <b>linha 179</b>: "
    "<code>torch.min(next_q1, next_q2)</code>) quanto a perda do ator na <b>linha 483</b> (<code>compute_actor_loss</code>, "
    "<b>linha 195</b>: <code>torch.min(q1_pi, q2_pi)</code>) calculam <i>min(Q<sub>1</sub>, Q<sub>1</sub>) = Q<sub>1</sub></i>. "
    "Como o ator move as ações na direção que maximiza <i>Q<sub>1</sub>(s, a)</i> e o próprio <i>Q'<sub>1</sub></i> avalia "
    "<i>a'</i> no alvo de Bellman, erros positivos de aproximação se retroalimentam continuamente.",
    body_style,
))

story.append(Image(str(FIGS / "p4_ablation_single_q.png"), width=540, height=143))

story.append(PageBreak())

# ==================== PÁGINA 3: PARTE 5 (SAC VS PPO CONTÍNUO) ====================
story.append(Paragraph("Parte 5 — SAC vs PPO Contínuo no Hopper-v5 (100 mil passos, 2 seeds)", h1_style))
story.append(Paragraph(
    "Comparamos o melhor SAC (<code>sac_hopper</code>, runs <code>ss0o8ouj</code> e <code>qbrrtmhs</code>, além de "
    "<code>sac_utd2</code>, runs <code>aebnfakf</code> e <code>f6l8uj5v</code>, para analisar <code>updates_per_step</code>) "
    "com o PPO contínuo (<code>ppo_hopper</code>, runs <code>q0z6trhx</code> e <code>5lsk4xhm</code>, rodados com "
    "<code>--config ppo_pendulum --override env_id=Hopper-v5 total_timesteps=100000 ppo_continuous_action.gamma=0.99 track=true seed=1,2</code>).",
    body_style,
))

story.append(Paragraph(
    "<b>Previsão (antes do experimento):</b> Esperávamos que, no orçamento de 100.000 passos, o SAC atingisse retorno "
    "muito superior por passo de ambiente graças ao reuso off-policy no replay buffer, mas que o PPO contínuo tivesse "
    "<code>charts/SPS</code> muito maior e <code>charts/num_updates</code> muito menor, terminando os 100k passos em poucos "
    "minutos e chegando mais longe no curtíssimo prazo de tempo de relógio.",
    body_style,
))

story.append(Image(str(FIGS / "p5_sac_vs_ppo.png"), width=540, height=122))
story.append(Spacer(1, 4))

story.append(Paragraph("1. Eficiência Amostral (Retorno × Passos) vs. Tempo de Relógio (Retorno × Minutos)", h2_style))
story.append(Paragraph(
    "&bull; <b>Eficiência amostral (painel a):</b> Confirmou a previsão. Em 100.000 passos, o PPO contínuo atinge "
    "<code>r_last100</code> de apenas <b>361.1</b> (<code>q0z6trhx</code>) e <b>320.7</b> (<code>5lsk4xhm</code>), com "
    "<code>eval/mean_return</code> médio de <b>370.6</b> (410.3 e 331.0). Com os mesmos 100.000 passos, os seeds 1 e 2 do "
    "<code>sac_hopper</code> atingem <code>r_last100</code> de <b>1294.2</b> e <b>862.2</b> (média <b>1078.2</b>; 1163.7 nos 3 seeds) "
    "e <code>eval/mean_return</code> médio de <b>2310.2</b> (2157.8 e 2462.5) — avaliação <b>6,2&times; maior</b>.<br/>"
    "&bull; <b>Tempo de relógio (painel b) — quem chega mais longe com o mesmo tempo?</b> Depende do horizonte de tempo: "
    "em <b>1,4 minutos</b> (tempo total em que o PPO conclui os 100k passos), <b>o PPO chega mais longe</b> (retorno 340.9, "
    "eval 370.6), pois em 1,4 min o <code>sac_hopper</code> (39 SPS) executou apenas ~3.300 passos e ainda está na coleta "
    "aleatória (<code>learning_starts=5000</code>, retorno ~18). Porém, após <b>~6 minutos</b> de relógio (~14k passos), "
    "o <code>sac_hopper</code> ultrapassa o retorno final do PPO e segue subindo até 42,1 minutos.<br/>"
    "&bull; <b>Explicação via <code>charts/SPS</code> (painel c) e <code>charts/num_updates</code> (painel d):</b> O PPO "
    "coleta rollouts vetorizados (<code>num_envs=4, num_steps=1024, batch_size=4096</code>) em 24 iterações e faz apenas "
    "24 &times; 10 &times; 32 = <b>7.680 passos de gradiente</b>, rodando a <b>1.246–1.247 SPS</b> (1,4 min). Já o "
    "<code>sac_hopper</code> (<code>updates_per_step=1</code>) faz 1 passo de gradiente dos críticos (<code>batch_size=256</code>) "
    "a cada passo do ambiente após o passo 5.000, somando <b>94.999 atualizações dos críticos</b> (<code>num_updates=94900</code> "
    "no log de 99.900) mais 47.499 de ator/&alpha;, reduzindo o throughput em 32&times; (<b>39–40 SPS</b>, 42,1 min).<br/>"
    "&bull; <b>Como <code>updates_per_step</code> da Parte 3 muda a resposta:</b> Com <code>sac.updates_per_step=2</code> "
    "(<code>sac_utd2</code>, verde nos painéis b–d), <code>charts/num_updates</code> dobra para <b>189.800</b>, "
    "<code>charts/SPS</code> cai exatamente à metade (<b>19–20 SPS</b>) e o tempo de relógio dobra para <b>83,3 min</b>, "
    "exigindo ~12 min (o dobro do <code>utd=1</code>) para superar o retorno que o PPO atinge em 1,4 min.",
    body_style,
))

story.append(Paragraph("2. Mecanismo e Estimativa de Passos de Gradiente por Transição", h2_style))
story.append(Paragraph(
    "&bull; <b>Peças do SAC que explicam as diferenças:</b> (1) <b>Replay buffer off-policy</b> (<code>ReplayBuffer</code>, "
    "linhas 212–243): retém todas as 100.000 transições (<code>buffer_size = 10<sup>6</sup></code>) para reuso contínuo, "
    "enquanto o PPO descarta o rollout de 4.096 transições ao fim de cada iteração; (2) <b>Ator reparametrizado + dois "
    "críticos Q</b> (linhas 145 e 195): atualiza o ator usando o gradiente determinístico <i>&nabla;<sub>&theta;</sub> "
    "min(Q<sub>1</sub>, Q<sub>2</sub>)(s, a<sub>&theta;</sub>(s))</i> em vez da razão de probabilidades ruidosa do PPO; "
    "(3) <b>Máxima entropia com &alpha; adaptativo</b> (linhas 179 e 206): evita colapso prematuro da política.<br/>"
    "&bull; <b>Quantas vezes, em média, cada transição entra em um passo de gradiente:</b>",
    body_style,
))

p5_rows = [
    [
        Paragraph("Algoritmo / Config", table_header),
        Paragraph("Passos de gradiente (num_updates)", table_header),
        Paragraph("Amostras por passo (batch_size)", table_header),
        Paragraph("Cálculo do reuso médio por transição em 100.000 passos", table_header),
        Paragraph("Reuso médio / transição", table_header),
        Paragraph("SPS / Tempo", table_header),
    ],
    [
        Paragraph("<b>PPO Contínuo</b><br/>(<code>ppo_hopper</code>)", table_cell),
        Paragraph("7.680<br/>(24 iter &times; 10 ép. &times; 32 mb)", table_cell),
        Paragraph("128<br/>(minibatch)", table_cell),
        Paragraph("Cada lote de 4.096 transições passa por <code>update_epochs = 10</code> épocas sem reposição e é descartado (98.304 passos usados).", table_cell),
        Paragraph("<b>10,0 vezes</b><br/>(9,83 sobre 100k)", table_cell),
        Paragraph("1.247 SPS<br/><b>1,4 min</b>", table_cell),
    ],
    [
        Paragraph("<b>SAC (utd = 1)</b><br/>(<code>sac_hopper</code>)", table_cell),
        Paragraph("94.999 Q updates<br/>(+47.499 ator/&alpha;)", table_cell),
        Paragraph("256", table_cell),
        Paragraph("<i>(T - L) &middot; u &middot; B / T</i> = 95.000 &times; 1 &times; 256 / 100.000 = <b>243,2</b> (transições iniciais <i>t &le; 5k</i>: 256 ln(100k/5k) &approx; <b>766,9 vezes</b>).", table_cell),
        Paragraph("<b>243,2 vezes</b><br/>(até 766,9 nas iniciais)", table_cell),
        Paragraph("39–40 SPS<br/><b>42,1 min</b>", table_cell),
    ],
    [
        Paragraph("<b>SAC (utd = 2)</b><br/>(<code>sac_utd2</code>)", table_cell),
        Paragraph("189.998 Q updates<br/>(+94.999 ator/&alpha;)", table_cell),
        Paragraph("256", table_cell),
        Paragraph("<i>(T - L) &middot; u &middot; B / T</i> = 95.000 &times; 2 &times; 256 / 100.000 = <b>486,4</b> (transições iniciais <i>t &le; 5k</i>: 512 ln(100k/5k) &approx; <b>1.533,8 vezes</b>).", table_cell),
        Paragraph("<b>486,4 vezes</b><br/>(até 1.533,8 nas iniciais)", table_cell),
        Paragraph("19–20 SPS<br/><b>83,3 min</b>", table_cell),
    ],
]

t_p5 = Table(p5_rows, colWidths=[78, 82, 58, 178, 82, 66])
t_p5.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7fafc"), colors.white]),
    ("TOPPADDING", (0, 0), (-1, -1), 3.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t_p5)

doc.build(story)
print(f"Successfully generated {PDF_PATH}")
