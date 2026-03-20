"""
Application Streamlit - Interface utilisateur pour la génération de fiches de cours
"""
import streamlit as st
from datetime import datetime
from pathlib import Path
import sys
from config import EDUCATION_LEVELS, SUPPORTED_SUBJECTS, OUTPUT_DIR  # PAS from .config
# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent))

from state import GraphState, InputData
try:
    from orchestrator import create_orchestrator
except ImportError as e:
    print(f"Erreur d'import: {e}")
    # Fallback : importer le module et accéder à la fonction
    import orchestrator
    create_orchestrator = orchestrator.create_orchestrator
from config import EDUCATION_LEVELS, SUPPORTED_SUBJECTS, OUTPUT_DIR


# Configuration de la page
st.set_page_config(
    page_title="Générateur de Fiches de Cours",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .section-header {
        background-color: #ecf0f1;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-size: 1.1rem;
        padding: 0.7rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialise l'état de la session"""
    if 'fiche_generee' not in st.session_state:
        st.session_state.fiche_generee = False
    if 'derniere_fiche' not in st.session_state:
        st.session_state.derniere_fiche = None


def get_classes_by_cycle(cycle):
    """Retourne les classes d'un cycle"""
    return EDUCATION_LEVELS.get(cycle, [])


def main():
    """Fonction principale de l'application"""
    init_session_state()
    
    # En-tête
    st.markdown('<h1 class="main-header"> Générateur de Fiches de Cours Multi-Agentsx   </h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; margin-bottom: 2rem;'>
        Système multi-agents intelligent pour la génération automatique de fiches de cours conformes aux programmes officiels
    </div>
    """, unsafe_allow_html=True)
    
    # Barre latérale avec informations
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.title("ℹ À propos")
        st.info("""
        **Système Multi-Agents**
        
        Ce générateur utilise 6 agents IA spécialisés :
        
        1.  **Agent Context** - Validation des données
        2.  **Agent Program** - Extraction des référentiels
        3.  **Agent Similarité** - Recherche de fiches existantes
        4.  **Agent Writer** - Génération du contenu (Ollama)
        5.  **Agent Validation** - Contrôle qualité
        6.  **Agent Export** - Export multi-formats
        """)
        
        st.markdown("---")
        
        st.success(f"""
        **Matières supportées (avec Corpus):**
        - {', '.join(SUPPORTED_SUBJECTS)}
        
        *Pour les autres matières, génération sans référentiel spécifique*
        """)
    
    # Formulaire principal
    st.markdown('<div class="section-header"><h2> Informations de la Fiche</h2></div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Informations de l'Établissement")
        
        etablissement = st.text_input(
            "Nom de l'établissement *",
            placeholder="Ex: Lycée Marie Curie",
            help="Nom complet de votre établissement scolaire"
        )
        
        ville = st.text_input(
            "Ville *",
            placeholder="Ex: Paris",
            help="Ville de l'établissement (utilisée pour l'ancrage local)"
        )
        
        annee_scolaire = st.text_input(
            "Année scolaire *",
            value=f"{datetime.now().year}-{datetime.now().year + 1}",
            help="Format: AAAA-AAAA"
        )
        
        nom_professeur = st.text_input(
            "Nom du professeur *",
            placeholder="Ex: M. Dupont",
            help="Votre nom complet"
        )
    
    with col2:
        st.subheader(" Informations Pédagogiques")
        
        # Sélection du cycle d'abord
        cycle = st.selectbox(
            "Cycle d'enseignement *",
            options=list(EDUCATION_LEVELS.keys()),
            help="Détermine le niveau général"
        )
        
        # Les classes disponibles dépendent du cycle
        classe = st.selectbox(
            "Classe *",
            options=get_classes_by_cycle(cycle),
            help="Niveau précis des élèves"
        )
        
        matiere = st.text_input(
            "Matière *",
            placeholder="Ex: Mathématiques, Informatique, Physique...",
            help="Matière enseignée (Informatique et Mathématiques ont un corpus dédié pour le Secondaire)"
        )
        
        theme_chapitre = st.text_input(
            "Thème/Chapitre *",
            placeholder="Ex: Les fonctions affines, L'érosion...",
            help="Sujet précis du cours"
        )
    
    # Informations complémentaires
    st.markdown('<div class="section-header"><h2> Paramètres du Cours</h2></div>', 
                unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        volume_horaire = st.number_input(
            "Volume horaire (heures) *",
            min_value=0.5,
            max_value=20.0,
            value=2.0,
            step=0.5,
            help="Durée totale du cours (détermine le gabarit)"
        )
    
    with col4:
        sequence_ou_date = st.text_input(
            "Séquence ou Date *",
            value=f"Séquence {datetime.now().month}",
            placeholder="Ex: Séquence 3, 15/01/2025",
            help="Numéro de séquence ou date du cours"
        )
    
    with col5:
        st.metric("Type de cours", 
                 "Court (1-2h)" if volume_horaire <= 2 
                 else "Moyen (3-4h)" if volume_horaire <= 4 
                 else "Étendu (5h+)")
    
    # Zone d'informations
    st.markdown("---")
    
    with st.expander("ℹ Informations importantes"):
        st.info("""
        **Génération avec Corpus (Secondaire uniquement):**
        - Pour les matières **Informatique** et **Mathématiques** au niveau **Secondaire**, 
          le système utilise les programmes officiels du dossier Corpus.
        
        **Situation-Problème:**
        - Générée automatiquement pour le niveau **Secondaire**
        - Ancrée dans le contexte local de votre ville
        
        **Processus de validation:**
        - Primaire: Conformité minimum 90%
        - Secondaire: Conformité minimum 85%
        - Universitaire: Conformité minimum 80%
        
        **Formats d'export:**
        - Markdown (.md) - Pour édition facile
        - JSON (.json) - Pour intégration système
        - HTML (.html) - Pour visualisation et impression
        """)
    
    # Bouton de génération
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        generer = st.button(" Générer la Fiche de Cours", type="primary")
    
    # Traitement de la génération
    if generer:
        # Validation des champs obligatoires
        champs_manquants = []
        
        if not etablissement:
            champs_manquants.append("Établissement")
        if not ville:
            champs_manquants.append("Ville")
        if not annee_scolaire:
            champs_manquants.append("Année scolaire")
        if not classe:
            champs_manquants.append("Classe")
        if not matiere:
            champs_manquants.append("Matière")
        if not nom_professeur:
            champs_manquants.append("Nom du professeur")
        if not theme_chapitre:
            champs_manquants.append("Thème/Chapitre")
        if not sequence_ou_date:
            champs_manquants.append("Séquence/Date")
        
        if champs_manquants:
            st.error(f" Champs obligatoires manquants: {', '.join(champs_manquants)}")
        else:
            # Créer l'input data
            input_data = InputData(
                etablissement=etablissement,
                ville=ville,
                annee_scolaire=annee_scolaire,
                classe=classe,
                volume_horaire=volume_horaire,
                matiere=matiere,
                nom_professeur=nom_professeur,
                theme_chapitre=theme_chapitre,
                sequence_ou_date=sequence_ou_date
            )
            
            # Créer l'état initial
            initial_state = GraphState(input_data=input_data)
            
            # Afficher la progression
            with st.spinner(" Génération en cours..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text(" Initialisation de l'orchestrateur...")
                    progress_bar.progress(10)
                    
                    # Créer l'orchestrateur
                    orchestrator = create_orchestrator()
                    
                    status_text.text(" Validation du contexte...")
                    progress_bar.progress(20)
                    
                    status_text.text(" Extraction du référentiel...")
                    progress_bar.progress(35)
                    
                    status_text.text(" Recherche de fiches similaires...")
                    progress_bar.progress(50)
                    
                    status_text.text(" Génération du contenu avec Ollama...")
                    progress_bar.progress(70)
                    
                    # Exécuter le workflow
                    final_state = orchestrator.run(initial_state)
                    
                    status_text.text(" Validation de la fiche...")
                    progress_bar.progress(85)
                    
                    status_text.text(" Export des fichiers...")
                    progress_bar.progress(95)
                    
                    progress_bar.progress(100)
                    status_text.text(" Génération terminée!")
                    
                    # Sauvegarder dans la session
                    st.session_state.fiche_generee = True
                    st.session_state.derniere_fiche = final_state
                    
                    # Afficher les résultats
                    st.markdown("---")
                    
                    if final_state.validation.valide:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3> Fiche Validée avec Succès!</h3>
                            <p><strong>Score de conformité:</strong> {final_state.validation.score_conformite}%</p>
                            <p><strong>Itérations:</strong> {final_state.compteur_boucles + 1}</p>
                            <p><strong>Mode de génération:</strong> {final_state.mode_generation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                            <h3> Fiche Exportée avec Réserves</h3>
                            <p><strong>Score de conformité:</strong> {final_state.validation.score_conformite}%</p>
                            <p><strong>Note:</strong> La fiche n'a pas atteint le seuil de validation mais a été exportée 
                            après {final_state.compteur_boucles + 1} itérations.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Afficher les fichiers générés
                    st.subheader(" Fichiers Générés")
                
                    output_files = list(OUTPUT_DIR.glob("fiche_*"))
                    latest_files = sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
                    
                    for file_path in latest_files:
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label=f" Télécharger {file_path.name}",
                                data=f,
                                file_name=file_path.name,
                                mime="application/octet-stream"
                            )
                    
                    # Aperçu de la fiche
                    st.markdown("---")
                    st.subheader(" Aperçu de la Fiche")
                    
                    with st.expander("Voir l'aperçu complet", expanded=True):
                        fiche = final_state.fiche
                        
                        st.markdown(f"### {fiche.titre}")
                        st.markdown(f"**Établissement:** {fiche.etablissement} - {fiche.ville}")
                        st.markdown(f"**Classe:** {fiche.classe}")
                        
                        st.markdown("#### Objectifs Pédagogiques")
                        for i, obj in enumerate(fiche.objectifs, 1):
                            st.markdown(f"{i}. {obj}")
                        
                        if fiche.situation_probleme:
                            st.markdown("#### Situation-Problème")
                            st.info(fiche.situation_probleme)
                        
                        st.markdown("#### Introduction")
                        st.write(fiche.introduction)
                        
                        st.markdown("#### Activités")
                        for i, act in enumerate(fiche.activites, 1):
                            st.markdown(f"**Activité {i}:** {act.get('titre', f'Activité {i}')}")
                            st.markdown(f"*Durée: {act.get('duree', 'Non spécifiée')}*")
                    
                    # Commentaires de validation
                    if final_state.validation.commentaires:
                        with st.expander(" Détails de la Validation"):
                            for comment in final_state.validation.commentaires:
                                st.markdown(f"- {comment}")

                    # === CORRECTION ICI ===
                    with st.expander("⏱️ Métriques de performance", expanded=False):
                        try:
                            summary = orchestrator.performance.get_summary()
                            
                            col_perf1, col_perf2, col_perf3 = st.columns(3)
                            
                            with col_perf1:
                                st.metric("Temps total", f"{summary['total_time']}s")
                                st.metric("Itérations", final_state.compteur_boucles + 1)
                            
                            with col_perf2:
                                st.metric("Score final", f"{final_state.validation.score_conformite}%")
                                st.metric("Statut", " Validé" if final_state.validation.valide else "⚠️ Réserves")
                            
                            with col_perf3:
                                st.metric("Agents exécutés", len(summary['agents']))
                            
                            # Détail par agent
                            st.markdown("##### ⏱️ Temps par agent (moyenne)")
                            agent_data = []
                            for agent_name, metrics in summary['agents'].items():
                                agent_data.append({
                                    "Agent": agent_name,
                                    "Temps moyen (s)": metrics['avg_time'],
                                    "Appels": metrics['calls']
                                })
                            
                            if agent_data:
                                st.dataframe(agent_data, use_container_width=True)
                        except AttributeError:
                            st.warning(" Les métriques de performance ne sont pas disponibles")
                    # === FIN CORRECTION ===

                except Exception as e:
                    st.error(f" Erreur lors de la génération: {str(e)}")
                    st.exception(e)


# Ajouter les métriques après la génération
if 'final_state' in st.session_state:
    with st.expander("📊 Métriques de performance", expanded=True):
        final_state = st.session_state.final_state
        if hasattr(orchestrator, 'performance'):
            summary = orchestrator.performance.get_summary()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Temps total", f"{summary['total_time']}s")
            col2.metric("Itérations", final_state.compteur_boucles + 1)
            col3.metric("Score", f"{final_state.validation.score_conformite}%")
            
            # Détail des agents
            st.subheader("Temps par agent")
            for agent, metrics in summary['agents'].items():
                st.text(f"{agent}: {metrics['avg_time']}s (x{metrics['calls']})")

if __name__ == "__main__":
    main()
