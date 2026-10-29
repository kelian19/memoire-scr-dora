"""Figures vectorielles de lecture du modèle, sans recalibration ni résultat nouveau."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyBboxPatch, Ellipse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'memoire/figures_pedagogie'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                     'font.size': 10, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})
INK = '#25333F'
RED = '#A52D39'


def arrow(ax, start, end, label=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12,
                                color=INK, linewidth=1.1))
    if label:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.12, label,
                ha='center', va='bottom', fontsize=9, color=INK)


def box(ax, xy, width, height, title, text, accent=False):
    x, y = xy
    ax.add_patch(Rectangle((x-width/2, y-height/2), width, height,
                           facecolor='#FBFBFA', edgecolor=RED if accent else INK, linewidth=1))
    ax.text(x, y+height*.23, title, ha='center', va='center', fontweight='bold',
            color=RED if accent else INK, fontsize=10)
    ax.text(x, y-height*.13, text, ha='center', va='center', fontsize=9, linespacing=1.45)


fig, ax = plt.subplots(figsize=(8.2, 3.9))
ax.set(xlim=(0, 12), ylim=(0, 6)); ax.axis('off')
box(ax, (6, 5.32), 9.5, .95, 'État de conformité supposé',
    r'Quatre paramètres : $\lambda$ (fréquence), $p_u$ (détection), $g$ (propagation), $\omega$ (choc commun)')
for x in [2,6,10]: arrow(ax,(x,4.83),(x,4.1))
box(ax,(2,3.45),3.3,1.25,"Combien d'incidents ?",r'$N$ : nombre annuel'+'\nAmorce : pilier de départ')
box(ax,(6,3.45),3.3,1.25,'Quels piliers sont touchés ?',r'$W$ : propagation dirigée'+'\n'+r'$S_a$ : ensemble atteint',True)
box(ax,(10,3.45),3.3,1.25,'Quel montant par pilier ?',r'$Y_{n,k}$ : perte élémentaire'+'\nZéro ou perte au-delà du seuil')
arrow(ax,(3.65,3.45),(4.35,3.45))
arrow(ax,(6,2.82),(6,2.16))
arrow(ax,(10,2.82),(8.45,1.75))
box(ax,(6,1.5),4.9,1.2,"Perte de l'incident, puis perte annuelle",
    r'$X_n=\sum_{k\in S_a}Y_{n,k}\ ;\quad L=\sum_{n=1}^{N}X_n$')
arrow(ax,(8.45,1.5),(9.1,1.5))
box(ax,(10.5,1.5),2.8,1.2,'Besoin de capital',r'$B=\mathrm{VaR}_{99,5\%}(L)$',True)
ax.text(.35,.35,"Les actifs de l'entité adaptent ensuite la fréquence et l'échelle des pertes ; le quantile est recalculé.",
        fontsize=9,va='center')
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
fig.savefig(OUT/'architecture_methode.pdf'); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 3.1))
ax.set(xlim=(0,12),ylim=(0,4.5)); ax.axis('off')
positions={4:(.8,3),1:(3.1,3),2:(5.4,3),3:(3.1,1.15),5:(5.4,1.15)}
for p,(x,y) in positions.items():
    touched=p in (4,1,2)
    ax.add_patch(Circle((x,y),.32,facecolor=INK if touched else 'white',edgecolor=INK,linewidth=1.2))
    ax.text(x,y,f'P{p}',color='white' if touched else INK,ha='center',va='center',fontweight='bold')
arrow(ax,(1.15,3),(2.75,3)); arrow(ax,(3.45,3),(5.05,3))
ax.text(2,3.3,'transmission',ha='center',fontsize=9)
ax.text(4.25,3.3,'transmission',ha='center',fontsize=9)
ax.text(.8,3.65,'Amorce',ha='center',fontsize=10,color=RED)
ax.text(5.4,3.65,'Arrêt',ha='center',fontsize=10,color=RED)
ax.text(4.25,.45,'P3 et P5 restent intacts',ha='center',fontsize=9)
ax.plot([6.25,6.25],[.4,4],color='#BBBBBB',linewidth=.8)
ax.text(6.85,3.65,r'$S_4=\{P4,P1,P2\}$',fontsize=13)
ax.text(6.85,2.9,r'$Y_4=a,\quad Y_1=b,\quad Y_2=c$',fontsize=12)
ax.text(6.85,2.25,r'$X_n=a+b+c$',fontsize=15,color=RED)
ax.text(6.85,1.4,'Un pilier touché est compté une fois.\nLa perte entre dans la somme de son année.',
        fontsize=10,linespacing=1.5)
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
fig.savefig(OUT/'incident_pas_a_pas.pdf'); plt.close(fig)

# Carte du raisonnement, distincte de la chaîne de simulation.
TEAL='#14828B'
def module(ax,x,y,w,h,title,lines,color=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.12',
                               facecolor='#FBFBFA',edgecolor=color,linewidth=1.25))
    ax.text(x+.16,y+h-.2,title,ha='left',va='top',fontsize=10.8,
            fontweight='bold',color=color,linespacing=1.2)
    ax.text(x+.16,y+h-.78,lines,ha='left',va='top',fontsize=9.5,linespacing=1.45)

# Le même fil de questions relie le calcul, sa mise à l'épreuve et son usage.
# Les apports bibliographiques sont détaillés dans le tableau de positionnement du chapitre 2.
GRIS_REPRIS='#5F676D'
def module_etape(ax,x,y,w,h,title,chapitres,question,methode,sortie,limite,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.12',
                               facecolor='#FBFBFA',edgecolor=color,linewidth=1.25))
    ax.text(x+.16,y+h-.16,title,ha='left',va='top',fontsize=10.6,fontweight='bold',color=color)
    ax.text(x+w-.16,y+h-.18,chapitres,ha='right',va='top',fontsize=8.6,color=GRIS_REPRIS)
    lignes=[(question,INK,'italic'),('Méthode : '+methode,INK,'normal'),
            ('Résultat : '+sortie,color,'normal'),('Portée : '+limite,GRIS_REPRIS,'normal')]
    yy=y+h-.62
    for texte,couleur,style in lignes:
        ax.text(x+.16,yy,texte,ha='left',va='top',fontsize=8.9,color=couleur,style=style,linespacing=1.3)
        yy-=.36*(texte.count('\n')+1)+.06

# 7,4 pouces et non 8,2 : incluse sur toute la largeur du texte, la figure garde ses textes
# vers 7,7 points imprimés au lieu de 5,7.
fig,ax=plt.subplots(figsize=(7.4,5.0))
ax.set(xlim=(0,12),ylim=(0,8.45));ax.axis('off')
ax.add_patch(FancyBboxPatch((2.6,7.6),6.8,.78,boxstyle='round,pad=.02,rounding_size=.25',
                            facecolor='#F3F5F7',edgecolor=GRIS_REPRIS,linewidth=1))
ax.text(6,8.13,'Données',ha='center',va='center',fontsize=10.2,fontweight='bold',color=INK)
ax.text(6,7.8,'SAS OpRisk · chronologie PRC · Hackmageddon · bilans SFCR',ha='center',va='center',fontsize=8.8)
module_etape(ax,.3,4.3,5.25,2.75,'1 · Définir le risque','ch. 1 à 3',
             'Quelles pertes et quelles données ?',
             'séparer montants, fréquences\net états de conformité',
             'entrées et hypothèses du calcul',
             'collectes incomplètes ; états supposés',INK)
module_etape(ax,6.45,4.3,5.25,2.75,'2 · Construire le calcul','ch. 4 à 6',
             "Comment un incident devient-il un capital ?",
             'compter, propager, chiffrer, sommer',
             'une loi annuelle par état supposé',
             'coûts par pilier et canaux posés',RED)
module_etape(ax,.3,.35,5.25,2.75,'3 · Éprouver les liens','ch. 7 et 8',
             'Peut-on mesurer qui entraîne qui ?',
             'confronter les sources ; W = S + A',
             'capital aux 1 024 sommets,\nforce commune des liens fixée',
             'direction non calibrée',TEAL)
module_etape(ax,6.45,.35,5.25,2.75,'4 · Comparer les choix','ch. 9 et 10',
             'Que peut-on raisonnablement décider ?',
             'fermer un canal ; varier les coûts\net la calibration',
             'gains conditionnels et incertitude',
             "validation d'entité encore nécessaire",INK)
arrow(ax,(4.2,7.57),(3.2,7.08)); arrow(ax,(7.8,7.57),(8.8,7.08))
ax.text(3.45,7.4,'Alimente',ha='right',fontsize=8.8,color=GRIS_REPRIS)
ax.text(8.55,7.4,'Calibre',ha='left',fontsize=8.8,color=GRIS_REPRIS)
arrow(ax,(5.58,5.7),(6.42,5.7)); ax.text(6,5.9,'Entrées',ha='center',fontsize=8.8)
arrow(ax,(2.9,4.27),(2.9,3.13)); ax.text(3.05,3.65,'Lacunes des sources',ha='left',fontsize=8.8)
arrow(ax,(9.1,4.27),(9.1,3.13)); ax.text(9.25,3.65,'Calculs',ha='left',fontsize=8.8)
arrow(ax,(5.58,1.75),(6.42,1.75)); ax.text(6,1.95,'Limites',ha='center',fontsize=8.8)
# Boucle de retour : le registre spécifié au chapitre 10 alimenterait de nouvelles données.
ax.plot([11.7,11.92,11.92],[1.3,1.3,8.0],color=GRIS_REPRIS,linewidth=.9,linestyle='--')
ax.add_patch(FancyArrowPatch((11.92,8.0),(9.45,8.0),arrowstyle='-|>',color=GRIS_REPRIS,
                             mutation_scale=10,linewidth=.9,linestyle='--'))
ax.text(10.68,8.15,"Registre d'incidents",ha='center',fontsize=8.6,color=GRIS_REPRIS)
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
fig.savefig(OUT/'architecture_memoire.pdf');plt.close(fig)

# Mécanisme opérationnel : scénario générique, pas une reconstitution d'incident public.
fig,ax=plt.subplots(figsize=(8.2,3.9))
ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
ax.text(.25,5.65,'UNE CAUSE COMMUNE',fontsize=11,fontweight='bold',color=INK)
ax.text(6.2,5.65,"AU SEIN D'UNE ENTITÉ",fontsize=11,fontweight='bold',color=RED)
module(ax,.4,3.35,3.0,1.45,'Prestataire partagé',
       'Service informatique\nindisponible',INK)
for x,y,lab in [(1,1.9,'Entité A'),(2.5,.8,'Entité B')]:
    ax.add_patch(Rectangle((x-.55,y-.35),1.4,.65,facecolor='white',edgecolor=INK))
    ax.text(x+.15,y,lab,ha='center',va='center',fontsize=10)
    arrow(ax,(1.9,3.32),(x+.15,y+.38))
arrow(ax,(3.43,4.05),(5.95,4.05))
ax.text(4.7,4.28,'Entité étudiée',fontsize=9,ha='center')
module(ax,6.15,3.35,5.45,1.45,'Services internes affectés',
       'Accès aux applications\nTraitement des opérations',INK)
arrow(ax,(8.85,3.32),(8.85,2.7))
module(ax,6.15,1.1,5.45,1.55,'Conséquences financières possibles',
       "Interruption d'activité · restauration des systèmes\nGestion de l'incident et reprise des opérations",RED)
ax.plot([5.7,5.7],[.5,5.35],color='#BBBBBB',linewidth=.8)
ax.text(.4,.05,'Un événement fournisseur peut toucher plusieurs entités.',fontsize=9)
ax.text(6.15,.35,'Contrôles : dépendance aux tiers, gestion des accès,\nconfinement et tests de reprise.',fontsize=9,linespacing=1.4)
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.03)
fig.savefig(OUT/'incident_et_controles.pdf');plt.close(fig)

# Même calcul sous deux jeux de paramètres : les capitaux restent symboliques.
fig,ax=plt.subplots(figsize=(8.2,3.65))
ax.set(xlim=(0,12),ylim=(0,5.5));ax.axis('off')
for y,title,col,vals,sub in [
    (3.8,'CONFORME',INK,('21,56','0,128','0,45','0'),r'C'),
    (1.45,'NON CONFORME',RED,('53,62','0,181','0,90','0,68'),r'NC')]:
    ax.text(.25,y+.95,title,fontweight='bold',fontsize=11,color=col)
    text=(r'$\lambda='+vals[0]+r'$ incidents/an ; '+r'$p_u='+vals[1]+r'$'
          +'\n'+r'$g='+vals[2]+r'$ ; '+r'$\omega='+vals[3]+r'$')
    module(ax,.25,y-.75,4.5,1.5,'Paramètres du scénario',text,col)
    arrow(ax,(4.8,y),(5.55,y))
    module(ax,5.6,y-.75,3.4,1.5,'Même moteur',
           'Incidents → piliers touchés\n→ somme annuelle',INK)
    arrow(ax,(9.05,y),(9.65,y))
    ax.text(10.7,y+.18,r'$L_{'+sub+r'}\;\longrightarrow\;B_{'+sub+r'}$',
            fontsize=13,ha='center',color=col)
    ax.text(10.7,y-.24,'Quantile annuel',fontsize=9,ha='center')
ax.plot([.25,11.7],[2.62,2.62],color='#CCCCCC',linewidth=.8,linestyle='--')
ax.text(.25,.22,"Les quatre écarts entre états sont posés. La loi de sévérité de base reste identique.",fontsize=9)
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
fig.savefig(OUT/'deux_etats_meme_calcul.pdf');plt.close(fig)
# Flux interne : les branches latérales apportent une entrée au calcul central.
def flux_bloc(ax,x,y,w,h,title,detail,color=INK):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
                 boxstyle='round,pad=.015,rounding_size=.06',edgecolor=color,
                 facecolor='#FAFBFC',linewidth=1))
    ax.text(x,y+h*.29,title,ha='center',va='center',fontsize=10.0,
            fontweight='bold',color=color)
    ax.text(x,y-h*.20,detail,ha='center',va='center',fontsize=9.5,linespacing=1.30)

fig,ax=plt.subplots(figsize=(8.2,5.2))
ax.set(xlim=(0,12),ylim=(0,10.2));ax.axis('off')
stages=[(9.45,'État de conformité fixé',r'$e$ règle $\lambda,\ p_u,\ g,\ \omega$',INK),
        (8.05,"Nombre d'incidents de l'année",r'$N_y$ : tirage binomial négatif',INK),
        (6.65,"Amorce de l'incident",r'$a_n$ : pilier de départ, poids posés',INK),
        (5.25,'Ensemble des piliers atteints',r'$S_{a_n}$ : propagation et règle de choc commun',INK),
        (3.85,"Perte de l'incident",r'$X_n=\sum_{k\in S_{a_n}}Y_{n,k}$',RED),
        (2.45,'Perte annuelle',r'$L_y=\sum_{n=1}^{N_y}X_n$',TEAL),
        (1.05,'Besoin de capital',r'$B_e=\mathrm{VaR}_{99,5\%}(L\mid e)$',INK)]
heights=[1.15 if title in ("Perte de l'incident",'Perte annuelle') else .92
         for _,title,_,_ in stages]
for (y,title,detail,col),height in zip(stages,heights):
    flux_bloc(ax,6,y,4.45,height,title,detail,col)
for j,((y,*_), (next_y,*__)) in enumerate(zip(stages,stages[1:])):
    arrow(ax,(6,y-heights[j]/2-.03),(6,next_y+heights[j+1]/2+.03))
flux_bloc(ax,1.55,8.05,2.8,1.25,'Loi de fréquence',r'$\lambda$ : moyenne annuelle'+'\n'+r'$\varphi$ : dispersion posée')
arrow(ax,(2.98,8.05),(3.75,8.05))
flux_bloc(ax,1.55,5.25,2.8,1.2,'Propagation',r'$T$ : forces des liens'+'\n'+r'$g,\omega$ : paramètres posés')
arrow(ax,(2.98,5.25),(3.75,5.25))
flux_bloc(ax,10.45,3.85,2.8,1.4,'Pertes élémentaires',r'$\mathbb{P}(Y_{n,k}=0)=1-p_u$'+'\n'+r'Sinon $u+\mathrm{GPD}(\xi,\sigma)$')
arrow(ax,(9.02,3.85),(8.25,3.85))
# Le nombre tiré détermine la somme : liaison de dépendance, pas de transformation.
ax.plot([8.27,8.6,8.6],[8.05,8.05,2.45],color='#81878C',linewidth=.85,linestyle='--')
ax.add_patch(FancyArrowPatch((8.6,2.45),(8.27,2.45),arrowstyle='-|>',
                           color='#81878C',mutation_scale=10,linewidth=.85))
ax.text(8.8,6.7,r'$N_y$ fixe le nombre',fontsize=8.8,ha='left',color='#626A70')
ax.text(8.8,6.35,"d'incidents à sommer",fontsize=8.8,ha='left',color='#626A70')
ax.text(.2,.12,'Trait plein : étape du calcul. Pointillés : quantité réutilisée. Les pertes sont exprimées en M€.',fontsize=9)
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.02)
fig.savefig(OUT/'flux_perte_annuelle.pdf');plt.close(fig)

# Deux horloges : la loi des états et le capital conditionnel restent des objets distincts.
fig,ax=plt.subplots(figsize=(8.2,4.05))
ax.set(xlim=(0,12),ylim=(0,6.2));ax.axis('off')
ax.add_patch(Rectangle((.15,1.65),5.35,4.35,facecolor='#F1F8F9',edgecolor='none'))
ax.add_patch(Rectangle((6.5,1.65),5.35,4.35,facecolor='#F3F5F7',edgecolor='none'))
ax.text(2.825,5.65,'CONFORMITÉ · MOIS ET ANNÉES',ha='center',fontsize=10.5,color=TEAL,fontweight='bold')
ax.text(9.175,5.65,'PERTES · UNE ANNÉE À ÉTAT FIXÉ',ha='center',fontsize=10.5,color=INK,fontweight='bold')
flux_bloc(ax,2.825,4.7,4.5,.85,'Entrées',r'$v_0$ : distribution ; $Q$ : générateur',TEAL)
arrow(ax,(2.825,4.25),(2.825,3.87))
flux_bloc(ax,2.825,3.4,4.5,.85,'Évolution des états',r'$v(t)=v_0\exp(Qt)$',TEAL)
arrow(ax,(2.825,2.95),(2.825,2.57))
ax.text(3.02,2.75,'Somme des phases',ha='left',va='center',fontsize=8.5,color=TEAL)
flux_bloc(ax,2.825,2.1,4.5,.85,'Probabilités au temps calendaire',r'$\mathbb{P}(e,t)$ pour NC, PC et C',TEAL)
flux_bloc(ax,9.175,4.7,4.5,.85,'Entrées conditionnelles à e',r'$\lambda(e),p_u(e),g(e),\omega(e)$',INK)
arrow(ax,(9.175,4.25),(9.175,3.87))
flux_bloc(ax,9.175,3.4,4.5,.85,'Incidents et pertes annuelles',r'$L\mid e$ : distribution simulée',INK)
arrow(ax,(9.175,2.95),(9.175,2.57))
flux_bloc(ax,9.175,2.1,4.5,.85,'Capital par état',r'$B_e=\mathrm{VaR}_{99,5\%}(L\mid e)$',INK)
ax.plot([6,6],[1.65,5.95],color=RED,linewidth=1,linestyle='--')
arrow(ax,(2.825,1.65),(2.825,1.15));arrow(ax,(9.175,1.65),(9.175,1.15))
flux_bloc(ax,6,.7,10.8,.85,'Trajectoire moyenne du besoin de capital',
          r'$\overline{B}(t)=\sum_e\mathbb{P}(e,t)\,B_e$',RED)
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
fig.savefig(OUT/'deux_horloges_conformite.pdf');plt.close(fig)
print('Sept figures pédagogiques vectorielles écrites dans',OUT)
