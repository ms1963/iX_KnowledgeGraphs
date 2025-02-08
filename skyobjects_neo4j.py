#!/usr/bin/env python
# -*- coding: utf-8 -*-

from py2neo import Graph
from transformers import pipeline
from functools import lru_cache
import logging
import sys
import time

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('astronomy_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Verbindung zur Neo4j-Graphdatenbank herstellen
try:
    graph = Graph("bolt://localhost:7687", auth=("username", "password"))
    logging.info("Erfolgreich mit Neo4j-Datenbank verbunden")
except Exception as e:
    logging.error(f"Fehler beim Verbinden mit der Datenbank: {str(e)}")
    sys.exit(1)

# Deutsches Frage-Antwort-Modell laden (besser geeignet für QA in Deutsch)
try:
    model_name = "deepset/gelectra-base-germanquad"
    qa_pipeline = pipeline(
        "question-answering",
        model=model_name,
        tokenizer=model_name,
        device=0  # falls GPU verfügbar ist, andernfalls: device=-1
    )
    logging.info("Frage-Antwort-Modell erfolgreich geladen")
except Exception as e:
    logging.error(f"Fehler beim Laden des Frage-Antwort-Modells: {str(e)}")
    sys.exit(1)

def query_graph_db(object_name):
    """Abfrage der Neo4j-Graphdatenbank mit einer sicheren parametrisierten Anfrage."""
    try:
        query = """
        MATCH (obj {name: $object_name})
        RETURN obj.name AS name, obj.type AS type, obj.distance_from_earth_ly AS distance
        """
        result = graph.run(query, object_name=object_name).data()
        return result[0] if result else None
    except Exception as e:
        logging.error(f"Datenbankabfrage-Fehler: {str(e)}")
        raise

@lru_cache(maxsize=1)
def get_available_objects():
    """Lädt alle verfügbaren Himmelsobjekte aus der Datenbank mit Caching."""
    try:
        query = """
        MATCH (obj)
        RETURN obj.name AS name
        ORDER BY obj.name
        """
        results = graph.run(query).data()
        return [result['name'] for result in results]
    except Exception as e:
        logging.error(f"Fehler beim Laden der verfügbaren Objekte: {str(e)}")
        raise

def extract_object_name(question):
    """Extrahiert Objektnamen auf robuste Weise durch Abgleich mit der Datenbank."""
    try:
        question_lower = question.lower()
        available_objects = get_available_objects()
        
        # Prüfen, ob eines der Objekte in der Frage vorkommt
        for obj in available_objects:
            if obj.lower() in question_lower:
                return obj
                
        return None
    except Exception as e:
        logging.error(f"Fehler bei der Objektnamenerkennung: {str(e)}")
        raise

def ask_question(question):
    """Verarbeitet eine Frage und generiert eine Antwort mithilfe des Frage-Antwort-Modells."""
    try:
        # Extrahiere den Objektnamen aus der Frage
        object_name = extract_object_name(question)
        
        if not object_name:
            return "Ich konnte kein bekanntes Himmelsobjekt in der Frage finden."
        
        # Hole Informationen aus der Datenbank
        info = query_graph_db(object_name)
        
        if info:
            # Erstelle einen verbesserten Kontext mit klaren Anweisungen
            context = (
                "Bitte beantworte die folgende Frage kurz und präzise basierend auf den nachfolgenden Fakten:\n\n"
                f"{info['name']} ist ein {info['type']}. "
                f"Es ist {info['distance']} Lichtjahre von der Erde entfernt."
            )
            
            # Verwende den QA-Pipeline: Übergabe von Frage und verbessertem Kontext
            result = qa_pipeline(question=question, context=context)
            return result["answer"]
        else:
            return "Ich habe keine Informationen zu diesem Himmelsobjekt."
    except Exception as e:
        logging.error(f"Fehler bei der Frageverarbeitung: {str(e)}")
        raise

def reset_cache():
    """Setzt den Cache für die verfügbaren Objekte zurück."""
    try:
        get_available_objects.cache_clear()
        logging.info("Cache wurde zurückgesetzt")
    except Exception as e:
        logging.error(f"Fehler beim Zurücksetzen des Caches: {str(e)}")
        raise

def display_help():
    """Zeigt Hilfe-Informationen an."""
    help_text = """
    Verfügbare Befehle:
    - exit: Beendet das Programm
    - help: Zeigt diese Hilfe an
    - update: Aktualisiert die Liste der verfügbaren Objekte
    - clear: Leert den Bildschirm
    
    Beispielfragen:
    - Wie weit ist [Objekt] von der Erde entfernt?
    - Was ist [Objekt]?
    - Beschreibe [Objekt].
    """
    print(help_text)

def main():
    """Hauptprogramm mit verbesserter Benutzerinteraktion und Fehlerbehandlung."""
    try:
        print("\n=== Astronomie-Informationssystem ===")
        print("Geben Sie 'help' ein für mehr Informationen.")
        
        # Dynamische Liste der verfügbaren Objekte laden
        available_objects = get_available_objects()
        print("\nVerfügbare Objekte:", ", ".join(available_objects))
        
        while True:
            try:
                user_input = input("\nIhre Frage: ").strip()
                
                # Befehlsverarbeitung
                if user_input.lower() == 'exit':
                    print("Auf Wiedersehen!")
                    break
                elif user_input.lower() == 'help':
                    display_help()
                    continue
                elif user_input.lower() == 'update':
                    reset_cache()
                    available_objects = get_available_objects()
                    print("Objektliste aktualisiert!")
                    print("Verfügbare Objekte:", ", ".join(available_objects))
                    continue
                elif user_input.lower() == 'clear':
                    print('\n' * 50)
                    continue
                
                # Frage verarbeiten
                start_time = time.time()
                answer = ask_question(user_input)
                processing_time = time.time() - start_time
                
                print("\nAntwort:", answer)
                logging.debug(f"Verarbeitungszeit: {processing_time:.2f} Sekunden")
                
            except Exception as e:
                print(f"\nFehler bei der Verarbeitung: {str(e)}")
                logging.error(f"Verarbeitungsfehler: {str(e)}")
                print("Bitte versuchen Sie es erneut oder geben Sie 'help' ein.")
                
    except Exception as e:
        logging.critical(f"Kritischer Fehler im Hauptprogramm: {str(e)}")
        print(f"\nEin kritischer Fehler ist aufgetreten: {str(e)}")
        print("Das Programm wird beendet.")
        sys.exit(1)

# Beispielanwendung
def run_example():
    """Führt ein Beispiel aus."""
    example_questions = [
        "Wie weit ist die Sonne von der Erde entfernt?",
        "Was ist der Orion-Nebel?",
        "Beschreibe die Andromeda-Galaxie."
    ]
    
    print("\n=== Beispielanwendung ===")
    for question in example_questions:
        print(f"\nFrage: {question}")
        try:
            answer = ask_question(question)
            print("Antwort:", answer)
        except Exception as e:
            print(f"Fehler bei der Beispielfrage: {str(e)}")

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--example":
            run_example()
        else:
            main()
    except KeyboardInterrupt:
        print("\nProgramm wurde vom Benutzer beendet.")
        logging.info("Programm durch Benutzer beendet")
    except Exception as e:
        print(f"\nEin unerwarteter Fehler ist aufgetreten: {str(e)}")
        logging.critical(f"Unerwarteter Fehler: {str(e)}")
    finally:
        logging.info("Programm beendet")
