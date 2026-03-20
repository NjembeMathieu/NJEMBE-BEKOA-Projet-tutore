"""
Agent Export - Générateur de Documents
Exporte la fiche validée en différents formats
"""
from pathlib import Path
from datetime import datetime
import json
from state import GraphState
from config import OUTPUT_DIR


class AgentExport:
    """Agent d'export des fiches validées"""
    
    def __init__(self):
        self.output_dir = OUTPUT_DIR
    
    def _generer_nom_fichier(self, state: GraphState) -> str:
        """Génère un nom de fichier unique"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        matiere = state.input_data.matiere.replace(" ", "_")
        classe = state.input_data.classe.replace(" ", "_")
        theme = state.input_data.theme_chapitre[:30].replace(" ", "_")
        
        return f"fiche_{matiere}_{classe}_{theme}_{timestamp}"
    
    def _formater_markdown(self, state: GraphState) -> str:
        """Formate la fiche en Markdown"""
        fiche = state.fiche
        input_data = state.input_data
        validation = state.validation
        
        md = f"""# {fiche.titre}

---

## Informations Générales

- **Établissement:** {fiche.etablissement}
- **Ville:** {fiche.ville}
- **Année scolaire:** {input_data.annee_scolaire}
- **Classe:** {fiche.classe}
- **Professeur:** {input_data.nom_professeur}
- **Matière:** {input_data.matiere}
- **Thème/Chapitre:** {input_data.theme_chapitre}
- **Séquence/Date:** {input_data.sequence_ou_date}
- **Volume horaire:** {input_data.volume_horaire}h

---

## Objectifs Pédagogiques

"""
        
        for i, objectif in enumerate(fiche.objectifs, 1):
            md += f"{i}. {objectif}\n"
        
        if fiche.situation_probleme:
            md += f"""
---

## Situation-Problème

{fiche.situation_probleme}

"""
        
        md += f"""---

## Introduction

{fiche.introduction}

---

## Développement du Cours

{fiche.developpement}

---

## Activités Pédagogiques

"""
        
        for i, activite in enumerate(fiche.activites, 1):
            md += f"""
### Activité {i}: {activite.get('titre', f'Activité {i}')}

**Durée:** {activite.get('duree', 'Non spécifiée')}

{activite.get('description', '')}

"""
        
        md += f"""---

## Évaluation

{fiche.evaluation}

---

## Conclusion

{fiche.conclusion}

"""
        
        if fiche.references:
            md += """---

## Références

"""
            for ref in fiche.references:
                md += f"- {ref}\n"
        
        # Ajouter les métadonnées de validation
        md += f"""
---

## Métadonnées de Validation

- **Score de conformité:** {validation.score_conformite}%
- **Statut:** {'✅ Validée' if validation.valide else '⚠️ Nécessite corrections'}
- **Mode de génération:** {state.mode_generation}
- **Nombre d'itérations:** {state.compteur_boucles + 1}

"""
        
        return md
    
    def _formater_json(self, state: GraphState) -> str:
        """Formate la fiche en JSON structuré"""
        fiche = state.fiche
        input_data = state.input_data
        validation = state.validation
        
        export_data = {
            "metadata": {
                "etablissement": input_data.etablissement,
                "ville": input_data.ville,
                "annee_scolaire": input_data.annee_scolaire,
                "classe": input_data.classe,
                "professeur": input_data.nom_professeur,
                "matiere": input_data.matiere,
                "theme_chapitre": input_data.theme_chapitre,
                "sequence_ou_date": input_data.sequence_ou_date,
                "volume_horaire": input_data.volume_horaire,
                "date_generation": datetime.now().isoformat()
            },
            "contenu": {
                "titre": fiche.titre,
                "objectifs": fiche.objectifs,
                "situation_probleme": fiche.situation_probleme,
                "introduction": fiche.introduction,
                "developpement": fiche.developpement,
                "activites": fiche.activites,
                "evaluation": fiche.evaluation,
                "conclusion": fiche.conclusion,
                "references": fiche.references
            },
            "validation": {
                "valide": validation.valide,
                "score_conformite": validation.score_conformite,
                "commentaires": validation.commentaires,
                "elements_manquants": validation.elements_manquants,
                "corrections_requises": validation.corrections_requises
            },
            "generation_info": {
                "mode": state.mode_generation,
                "iterations": state.compteur_boucles + 1,
                "historique_corrections": state.historique_corrections
            }
        }
        
        return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def _formater_html(self, state: GraphState) -> str:
        """Formate la fiche en HTML"""
        fiche = state.fiche
        input_data = state.input_data
        validation = state.validation
        
        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{fiche.titre}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        .objectifs, .activites {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .situation-probleme {{
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
        .activite {{
            background-color: white;
            padding: 15px;
            margin: 10px 0;
            border-left: 3px solid #3498db;
        }}
        .validation {{
            background-color: {'#d4edda' if validation.valide else '#f8d7da'};
            border: 1px solid {'#c3e6cb' if validation.valide else '#f5c6cb'};
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }}
        ul {{
            margin: 10px 0;
        }}
        li {{
            margin: 5px 0;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            background-color: #3498db;
            color: white;
            border-radius: 3px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{fiche.titre}</h1>
        
        <div class="metadata">
            <p><strong>Établissement:</strong> {fiche.etablissement}</p>
            <p><strong>Ville:</strong> {fiche.ville}</p>
            <p><strong>Année scolaire:</strong> {input_data.annee_scolaire}</p>
            <p><strong>Classe:</strong> {fiche.classe}</p>
            <p><strong>Professeur:</strong> {input_data.nom_professeur}</p>
            <p><strong>Matière:</strong> {input_data.matiere}</p>
            <p><strong>Thème/Chapitre:</strong> {input_data.theme_chapitre}</p>
            <p><strong>Séquence/Date:</strong> {input_data.sequence_ou_date}</p>
            <p><strong>Volume horaire:</strong> {input_data.volume_horaire}h</p>
        </div>
        
        <h2>Objectifs Pédagogiques</h2>
        <div class="objectifs">
            <ul>
"""
        
        for objectif in fiche.objectifs:
            html += f"                <li>{objectif}</li>\n"
        
        html += """            </ul>
        </div>
"""
        
        if fiche.situation_probleme:
            html += f"""
        <h2>Situation-Problème</h2>
        <div class="situation-probleme">
            <p>{fiche.situation_probleme}</p>
        </div>
"""
        
        html += f"""
        <h2>Introduction</h2>
        <p>{fiche.introduction}</p>
        
        <h2>Développement du Cours</h2>
        <div>{fiche.developpement}</div>
        
        <h2>Activités Pédagogiques</h2>
        <div class="activites">
"""
        
        for i, activite in enumerate(fiche.activites, 1):
            html += f"""
            <div class="activite">
                <h3>Activité {i}: {activite.get('titre', f'Activité {i}')}</h3>
                <p><span class="badge">Durée: {activite.get('duree', 'Non spécifiée')}</span></p>
                <p>{activite.get('description', '')}</p>
            </div>
"""
        
        html += f"""
        </div>
        
        <h2>Évaluation</h2>
        <p>{fiche.evaluation}</p>
        
        <h2>Conclusion</h2>
        <p>{fiche.conclusion}</p>
"""
        
        if fiche.references:
            html += """
        <h2>Références</h2>
        <ul>
"""
            for ref in fiche.references:
                html += f"            <li>{ref}</li>\n"
            html += """        </ul>
"""
        
        html += f"""
        <div class="validation">
            <h3>Validation de la Fiche</h3>
            <p><strong>Score de conformité:</strong> {validation.score_conformite}%</p>
            <p><strong>Statut:</strong> {'✅ Validée' if validation.valide else '⚠️ Nécessite corrections'}</p>
            <p><strong>Mode de génération:</strong> {state.mode_generation}</p>
            <p><strong>Nombre d'itérations:</strong> {state.compteur_boucles + 1}</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def process(self, state: GraphState) -> GraphState:
        """
        Exporte la fiche dans différents formats
        """
        # Générer le nom de base
        base_name = self._generer_nom_fichier(state)
        
        # Exporter en Markdown
        md_content = self._formater_markdown(state)
        md_path = self.output_dir / f"{base_name}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        # Exporter en JSON
        json_content = self._formater_json(state)
        json_path = self.output_dir / f"{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        # Exporter en HTML
        html_content = self._formater_html(state)
        html_path = self.output_dir / f"{base_name}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Fiche exportée:")
        print(f"   - Markdown: {md_path}")
        print(f"   - JSON: {json_path}")
        print(f"   - HTML: {html_path}")
        
        return state


def agent_export_node(state: GraphState) -> GraphState:
    """Node LangGraph pour l'Agent Export"""
    agent = AgentExport()
    return agent.process(state)