from typing import Dict, List

from neo4j import GraphDatabase

from .config import settings
from .stores import get_mongo


def get_driver():
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


def seed_graph_from_mongo():
    db = get_mongo()
    documents = list(db[settings.mongo_docs_collection].find({}, {'_id': 0}))
    driver = get_driver()
    with driver.session() as session:
        session.run('CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE')
        session.run('CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE')
        session.run('CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE')
        session.run('MATCH (n) DETACH DELETE n')
        for doc in documents:
            session.run(
                '''
                MERGE (p:Paper {paper_id:$paper_id})
                SET p.title=$title, p.filename=$filename, p.year=$year, p.venue=$venue
                ''', **doc
            )
            for author in doc.get('authors', []) or ['Unknown Author']:
                session.run(
                    '''
                    MERGE (a:Author {name:$author})
                    WITH a
                    MATCH (p:Paper {paper_id:$paper_id})
                    MERGE (a)-[:WROTE]->(p)
                    ''', author=author, paper_id=doc['paper_id']
                )
            for topic in doc.get('topics', []) or ['general ai']:
                session.run(
                    '''
                    MERGE (t:Topic {name:$topic})
                    WITH t
                    MATCH (p:Paper {paper_id:$paper_id})
                    MERGE (p)-[:ABOUT]->(t)
                    ''', topic=topic, paper_id=doc['paper_id']
                )
            if doc.get('venue'):
                session.run(
                    '''
                    MERGE (v:Venue {name:$venue})
                    WITH v
                    MATCH (p:Paper {paper_id:$paper_id})
                    MERGE (p)-[:PUBLISHED_IN]->(v)
                    ''', venue=doc.get('venue'), paper_id=doc['paper_id']
                )
    driver.close()
    return {'papers': len(documents)}


def cypher_query(query: str, parameters: Dict = None) -> List[Dict]:
    parameters = parameters or {}
    driver = get_driver()
    with driver.session() as session:
        rows = [r.data() for r in session.run(query, **parameters)]
    driver.close()
    return rows


EXAMPLE_CYPHER_QUERIES = [
    {
        'name': 'Count nodes by label',
        'cypher': 'MATCH (n) RETURN labels(n) AS labels, count(n) AS count ORDER BY count DESC'
    },
    {
        'name': 'List papers with topics',
        'cypher': 'MATCH (p:Paper)-[:ABOUT]->(t:Topic) RETURN p.title AS paper, collect(t.name) AS topics LIMIT 10'
    },
    {
        'name': 'Find authors for a paper title keyword',
        'cypher': "MATCH (a:Author)-[:WROTE]->(p:Paper) WHERE toLower(p.title) CONTAINS toLower($keyword) RETURN p.title AS paper, collect(a.name) AS authors LIMIT 10",
        'parameters': {'keyword': 'learning'}
    },
    {
        'name': 'Find papers about a topic keyword',
        'cypher': "MATCH (p:Paper)-[:ABOUT]->(t:Topic) WHERE toLower(t.name) CONTAINS toLower($topic) RETURN t.name AS topic, p.title AS paper LIMIT 10",
        'parameters': {'topic': 'retrieval'}
    },
    {
        'name': 'Find papers sharing the same topic',
        'cypher': 'MATCH (p1:Paper)-[:ABOUT]->(t:Topic)<-[:ABOUT]-(p2:Paper) WHERE p1.paper_id <> p2.paper_id RETURN p1.title AS paper_a, t.name AS shared_topic, p2.title AS paper_b LIMIT 10'
    },
]
