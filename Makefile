# Commandes courantes du projet. Sur Windows sans make, lancer directement la commande
# Python indiquee sous chaque cible (voir CONTRIBUTING.md).

PY ?= python
MEMOIRE := exploratory/memoire_cascade

.PHONY: aide verifier institut memoire depot audit

aide: ## liste les cibles
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

verifier: ## rejoue le harnais : chaque nombre publie confirme par la sortie de son script
	$(PY) ci/harnais_strict.py

institut: ## regenere main_institut.tex depuis main_ensae.tex
	$(PY) $(MEMOIRE)/build_institut.py

memoire: ## compile les deux versions et passe tous les controles, sans le harnais
	$(PY) $(MEMOIRE)/refaire_tout.py --sans-harnais

depot: ## chaine complete : generer, compiler, controler, recopier les fichiers deposes
	$(PY) $(MEMOIRE)/refaire_tout.py

audit: ## cherche les correspondances fortuites du harnais (sources a relire a la main)
	$(PY) $(MEMOIRE)/audit_sources.py sorties_verif
