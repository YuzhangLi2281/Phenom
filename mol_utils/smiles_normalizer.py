from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import threading

from rdkit import Chem
from rdkit.Chem import AllChem
try:
    from papyrus_structure_pipeline import standardizer as Papyrus_standardizer
except ModuleNotFoundError:
    Papyrus_standardizer = None

T = TypeVar('T')

class TimeoutHandler:
    """Clean timeout handler that can be used with any function."""
    
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds
        
    def run_with_timeout(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
        """Run function with timeout."""
        outcome: dict[str, Any] = {}

        def worker():
            try:
                outcome["result"] = func(*args, **kwargs)
            except Exception as e:
                outcome["exception"] = e
        
        thread = threading.Thread(target=worker)
        thread.daemon = True  # Allow program to exit if thread is still running
        
        thread.start()
        thread.join(timeout=self.timeout_seconds)
        
        if thread.is_alive():
            return None  # Timeout occurred
        if "exception" in outcome:
            raise outcome["exception"]
        return outcome.get("result")

@dataclass
class NormalizationResult:
    smiles: str | None
    status: str
    error_message: str | None = None

class MoleculeNormalizer:
    """Clean molecule normalization class with timeout handling."""
    
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_handler = TimeoutHandler(timeout_seconds)
        
    def _standardize_molecule(self, mol):
        """Internal standardization function."""
        if Papyrus_standardizer is None:
            return None
        return Papyrus_standardizer.standardize(
            mol,
            filter_inorganic=False,
            small_molecule_min_mw=100,
            small_molecule_max_mw=2000,
            return_type=True
        )
    
    def standardize_normalize(self, smiles: str) -> NormalizationResult:
        """Normalize a SMILES string with timeout handling."""
        try:
            # Convert SMILES to mol
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return NormalizationResult(None, "INVALID_SMILES", "Could not parse SMILES")

            if Papyrus_standardizer is None:
                fallback = self.normalize(smiles)
                if fallback.smiles is not None:
                    fallback.status = "SUCCESS_NO_CHEMBL_STD"
                return fallback
            
            # Run standardization with timeout
            result = self.timeout_handler.run_with_timeout(self._standardize_molecule, mol)
            if result is None:
                return NormalizationResult(None, "TIMEOUT", "Standardization timeout")
            
            # Process standardization result
            out_type = result[1].value
            status_mapping = {
                1: ("SUCCESS", None),
                2: ("NON_SMALL_MOLECULE", "Molecular weight outside allowed range"),
                3: ("INORGANIC_MOLECULE", "Molecule contains non-organic elements"),
                4: ("MIXTURE", "Multiple fragments detected"),
                5: ("STANDARDIZATION_ERROR", "General standardization error")
            }
            
            if out_type != 1:
                status, message = status_mapping.get(
                    out_type,
                    ("UNKNOWN_ERROR", f"Unknown standardization result type: {out_type}")
                )
                return NormalizationResult(None, status, message)
            
            # Post-process
            mol = result[0]
            mol = Chem.RemoveHs(mol, sanitize=True)
            mol = AllChem.RemoveAllHs(mol)
            normalized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return NormalizationResult(normalized_smiles, "SUCCESS", None)
            
        except Exception as e:
            return NormalizationResult(None, "GENERAL_ERROR", str(e))
    
    def normalize(self, smiles: str) -> NormalizationResult:
        """Normalize a SMILES string with timeout handling."""
        try:
            # Convert SMILES to mol
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return NormalizationResult(None, "INVALID_SMILES", "Could not parse SMILES")
    
            mol = Chem.RemoveHs(mol, sanitize=True)
            mol = AllChem.RemoveAllHs(mol)

            normalized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            if normalized_smiles:
                return NormalizationResult(normalized_smiles, "SUCCESS_NO_STD", None)
            return NormalizationResult(None, "GENERAL_ERROR", "SMILES canonicalization returned an empty value")
        except Exception as e:
            return NormalizationResult(None, "GENERAL_ERROR", str(e))
    
