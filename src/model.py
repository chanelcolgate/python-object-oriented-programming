from typing import (
        Iterator,
        cast,
        Optional,
        List,
        Iterable,
        Protocol,
        Tuple,
        List,
        Counter)
from pathlib import Path
import abc
import collections
import csv
import enum

import weakref
import datetime

class InvalidSampleError(ValueError):
    """Source data file has invalid data representation"""
    pass


## class Sample:
##     def __init__(
##             self,
##             sepal_length: float,
##             sepal_width: float,
##             petal_length: float,
##             petal_width: float,
##             species: Optional[str] = None
##     ) -> None:
##         self.sepal_length = sepal_length
##         self.sepal_width = sepal_width
##         self.petal_length = petal_length
##         self.petal_width = petal_width
##         self.species = species
##         self.classification: Optional[str] = None
## 
##     def __repr__(self) -> str:
##         if self.species is None:
##             known_unknown = "UnknownSample"
##         else:
##             known_unknown = "KnownSample"
##         if self.classification is None:
##             classification = ""
##         else:
##             classification = f", classification={self.classification!r}"
## 
##         return (
##                 f"{known_unknown}("
##                 f"sepal_length={self.sepal_length}, "
##                 f"sepal_width={self.sepal_width}, "
##                 f"petal_length={self.petal_length}, "
##                 f"petal_width={self.petal_width}, "
##                 f"species={self.species!r}"
##                 f"{classification}"
##                 f")"
##         )
## 
##     def classify(self, classification: str) -> None:
##         self.classification = classification
## 
##     def matches(self) -> bool:
##         return self.species == self.classification

class Sample:
    """Abstract superclass for all samples."""
    def __init__(
            self,
            sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float,
    ) -> None:
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    ## def __repr__(self) -> str:
    ##     return (
    ##             f"{self.__class__.__name__}("
    ##             f"sepal_legnth={self.sepal_length}, "
    ##             f"sepal_width={self.sepal_width}, "
    ##             f"petal_length={self.petal_length}, "
    ##             f"petal_width={self.petal_width}, "
    ##             f")"
    ##     )

    @property
    def attr_dict(self) -> dict[str, str]:
        return dict(
                sepal_length=f"{self.sepal_length!r}",
                sepal_width=f"{self.sepal_width!r}",
                petal_length=f"{self.petal_length!r}",
                petal_width=f"{self.petal_width!r}",
        )

    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"

class Purpose(enum.IntEnum):
    Classification = 0
    Testing = 1
    Training = 2

class KnownSample(Sample):
    """
    Represents a sample of testing or training data, the species is set once
    The purpose determines if it can or cannot be classified
    """
    def __init__(
            self,
            species: str,
            purpose: int,
            sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float,
    ) -> None:
        super().__init__(
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width,
        )
        self.species = species
        purpose_enum = Purpose(purpose)
        if purpose_enum not in {Purpose.Training, Purpose.Testing}:
            raise ValueError(f"invalid prupose: {purpose!r}: {purpose_enum}")
        self.purpose = purpose_enum
        self._classification: Optional[str] = None

    def matches(self) -> bool:
        return self.species == self.classification

    @property
    def classification(self) -> Optional[str]:
        if self.purpose == Purpose.Testing:
            return self._classification
        else:
            raise AttributeError(f"Training samples have no classification")

    @classification.setter
    def classification(self, value: str) -> None:
        if self.purpose == Purpose.Testing:
            self._classification = value
        else:
            raise AttributeError(f"Training samples cannot be classified")

    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        base_attributes["purpose"] = f"{self.purpose.value}"
        base_attributes["species"] = f"{self.species!r}"
        if self.purpose == Purpose.Testing and self._classification:
            base_attributes["classification"] = f"{self.classification!r}"
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"

class UnknownSample(Sample):
    """A sample provided by a User, not yet classifier."""
    def __init__(
            self,
            sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float,
    ) -> None:
        super().__init__(
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width,
        )
        self._classification: Optional[str] = None

    @property
    def classification(self) -> Optional[str]:
        return self._classification

    @classification.setter
    def classification(self, value: str) -> None:
        self._classification = value

    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        base_attributes["classification"] = f"{self.classification!r}"
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"

class ClassifiedSample(Sample):
    """Created from a sample provided by a User, and the results of classification."""
    def __init__(
            self,
            classification: str,
            sample: UnknownSample
    ) -> None:
        super().__init__(
                sepal_length=sample.sepal_length,
                sepal_width=sample.sepal_width,
                petal_length=sample.petal_length,
                petal_width=sample.petal_width,
        )
        self.classification = classification

    def __repr__(self) -> str:
        return (
                f"{self.__class__.__name__}("
                f"sepal_length={self.sepal_length}, "
                f"sepal_width={self.sepal_width}, "
                f"petal_length={self.petal_length}, "
                f"petal_width={self.petal_width}, "
                f"classification={self.classification!r}, "
                f")"
        )

class Distance:
    """A distance computation"""
    def distance(self, s1: Sample, s2: Sample) -> float:
        raise NotImplementedError

class Chebyshev(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
                [
                    abs(s1.sepal_length - s2.sepal_length),
                    abs(s1.sepal_width - s2.sepal_width),
                    abs(s1.petal_length - s2.petal_length),
                    abs(s1.petal_width - s2.petal_width),
                ]
                )

class Minkowski(Distance):
    """An abstraction to provide a way to implement Manhattan and Euclidean."""
    m: int

    def distance(self, s1: Sample, s2: Sample) -> float:
        return (
                sum(
                    [
                        abs(s1.sepal_length - s2.sepal_length) ** self.m,
                        abs(s1.sepal_width - s2.sepal_width) ** self.m,
                        abs(s1.petal_length - s2.petal_length) ** self.m,
                        abs(s1.petal_width - s2.petal_width) ** self.m,
                    ]
                )
                ** (1 / self.m)
                )

class Euclidean(Minkowski):
    m = 2

class Mahattan(Minkowski):
    m = 1

class Sorensen(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
                [
                    abs(s1.sepal_length - s2.sepal_length),
                    abs(s1.sepal_width - s2.sepal_width),
                    abs(s1.petal_length - s2.petal_length),
                    abs(s1.petal_width - s2.petal_width),
                ]
                ) / sum (
                [
                    s1.sepal_length + s2.sepal_length,
                    s1.sepal_width + s2.sepal_width,
                    s1.petal_length + s2.petal_length,
                    s1.petal_width + s2.petal_width,
                ]
                )

## class Reduce_Function(Protocol):
##     """Define a callable object with specific parameters."""
##     def __call__(self, values: list[float]) -> float:
##         pass
## 
## class Minkowski_2(Distance):
##     """A generic way to implement Manhattan, Euclidean, and Chebyshev."""
##     m: int
##     reduction: Reduce_Function
##     def distance(self, s1: Sample, s2: Sample) -> float:
##         # Required to prevent Python from passing `self` as the first argument.
##         summarize = self.reduction
##         return (
##             summarize(
##                 [
##                     abs(s1.sepal_length - s2.sepal_length) ** self.m,
##                     abs(s1.sepal_width - s2.sepal_width) ** self.m,
##                     abs(s1.petal_length - s2.petal_length) ** self.m,
##                     abs(s1.petal_width - s2.petal_width) ** self.m,
##                 ]
##             )
##             ** (1 / self.m)
##         )
## 
## class ED2S(Minkowski_2):
##     m = 2
##     reduction = sum

class Minkowski_2(Distance):
    @property
    @abc.abstractmethod
    def m(self) -> int:
        ...

    @staticmethod
    @abc.abstractstaticmethod
    def reduction(values: Iterable[float]) -> float:
        ...

    def distance(self, s1: Sample, s2: Sample) -> float:
        return (
            self.reduction(
                [
                    abs(s1.sepal_length - s2.sepal_length) ** self.m,
                    abs(s1.sepal_width - s2.sepal_width) ** self.m,
                    abs(s1.petal_length - s2.petal_length) ** self.m,
                    abs(s1.petal_width - s2.petal_width) ** self.m,
                ]
            )
            ** (1 / self.m)
        )

class CD2(Minkowski_2):
    m = 1
    
    @staticmethod
    def reduction(values: Iterable[float]) -> float:
        return max(values)

class MD2(Minkowski_2):
    m = 1

    @staticmethod
    def reduction(values: Iterable[float]) -> float:
        return sum(values)

class ED2(Minkowski_2):
    m = 2

    @staticmethod
    def reduction(values: Iterable[float]) -> float:
        return sum(values)

class Hyperparameter:
    """
    A hyperparameter value and the overall quality of the classification.
    """
    def __init__(self, k: int, training: "TrainingData", algorithm: "Distance") -> None:
        self.k = k
        self.algorithm = algorithm
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality: float

    def test(self) -> None:
        """Run the entire test suite."""
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError("Borken weak Reference")
        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, sample: Sample) -> str:
        """TODO: the k-NN algorithm"""
        training_data = self.data()
        if not training_data:
            raise RuntimeError("No TrainingData object")
        distances: List[Tuple[float, KnownSample]] = sorted(
                (self.algorithm.distance(sample, known), known)
                for known in training_data.training
        )
        k_nearest = (known.species for d, known in distances[:self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common()
        species, votes = best_fit
        return species

class TrainingData:
    """
    A set of training data and testing data with methods to load and test the samples.
    """
    def __init__ (self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: List[KnownSample] = []
        self.testing: List[KnownSample] = []
        self.tuning: List[Hyperparameter] = []

    ## def load(
    ##         self,
    ##         raw_data_iter: Iterable[dict[str, str]]
    ## ) -> None:
    ##     """Extract TestingKnownSample and TrainingKnownSample from raw data"""
    ##     bad_count = 0
    ##     for n, row in enumerate(raw_data_iter):
    ##         try:
    ##             if n % 5 == 0:
    ##                 test = TestingKnownSample.from_dict(row)
    ##                 self.testing.append(test)
    ##             else:
    ##                 train = TrainingKnownSample.from_dict(row)
    ##                 self.training.append(train)
    ##         except InvalidSampleError as ex:
    ##             print(f"Row {n+1}: {ex}")
    ##             bad_count += 1
    ##     if bad_count != 0:
    ##         print(f"{bad_count} invalid rows")
    ##         return
    ##     self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)
    
    def load(
            self,
            raw_data_iter: Iterable[dict[str,str]]
    ) -> None:
        """Extract TestingKnownSample and TrainingKnownSample from raw data"""
        for n, row in enumerate(raw_data_iter):
            purpose = Purpose.Testing if n % 5 == 0 else Purpose.Training
            sample = KnownSample(
                    sepal_length=float(row["sepal_length"]),
                    sepal_width=float(row["sepal_width"]),
                    petal_length=float(row["petal_length"]),
                    petal_width=float(row["petal_width"]),
                    purpose=purpose,
                    species=row["species"],
            )
            if sample.purpose == Purpose.Testing:
                self.testing.append(sample)
            else:
                self.training.append(sample)
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(
            self,
            parameter: Hyperparameter) -> None:
        """Test this Hyperparameter value."""
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(
            self,
            parameter: Hyperparameter,
            sample: UnknownSample
    ) -> str:
        return parameter.classify(sample)

class BadSampleRow(ValueError):
    pass

class SampleReader:
    """
    See iris.names for attribute ordering in bezdekIris.data file
    """
    target_class = Sample
    header = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    def __init__(self, source: Path) -> None:
        self.source = source

    def sample_iter(self) -> Iterator[Sample]:
        target_class = self.target_class
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            for row in reader:
                try:
                    sample = target_class(
                            sepal_length=float(row['sepal_length']),
                            sepal_width=float(row['sepal_width']),
                            petal_length=float(row['petal_length']),
                            petal_width=float(row['petal_width']),
                    )
                except ValueError as ex:
                    raise BadSampleRow(f'invalid {row!r}') from ex
                yield sample
