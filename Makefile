# Commandes courantes du projet. Sur Windows sans make, lancer directement la commande
# Python indiquee sous chaque cible (voir CONTRIBUTING.md).

PY ?= python
.PHONY: aide etat structure verifier memoire depot audit

aide: ## liste les cibles
	$(PY) projet.py --help

etat: ## branche, changements locaux et PDF presents
	$(PY) projet.py etat

structure: ## chemins, liens des guides et fichiers suivis
	$(PY) projet.py structure

verifier: ## controle la structure puis rejoue le harnais strict
	$(PY) projet.py verifier

memoire: ## compile les deux versions et passe tous les controles, sans le harnais
	$(PY) projet.py compiler

depot: ## controles stricts, compilation et copies des fichiers de depot
	$(PY) projet.py depot

audit: ## cherche les correspondances fortuites du harnais (sources a relire a la main)
	$(PY) projet.py audit
