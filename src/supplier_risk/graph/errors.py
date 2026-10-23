"""Typed errors for the Module 4 Neo4j graph store.

The engine raises project errors rather than leaking driver exceptions, and the
assignment-facing wrapper converts them into tagged ``[GRAPH ERROR]`` strings.
"""

from __future__ import annotations


class GraphError(Exception):
    """Base class for graph configuration, connection, query, and data errors."""


class GraphConfigError(GraphError):
    """Raised when Neo4j settings are missing or cannot be used to connect."""


class GraphConnectionError(GraphError):
    """Raised when the Neo4j server cannot be reached or the session expires."""


class GraphQueryError(GraphError):
    """Raised when Neo4j rejects a query or the query exceeds its timeout."""


class GraphDataIntegrityError(GraphError):
    """Raised when ownership data or an existing graph conflicts with the dataset."""


class UnknownGraphEntityError(GraphError):
    """Raised when a supplier identity is unknown or ambiguous in the graph."""
