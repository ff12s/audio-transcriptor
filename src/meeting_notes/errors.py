class MeetingNotesError(Exception):
    """Ожидаемая ошибка, которую CLI показывает без стектрейса."""


class CudaUnavailableError(MeetingNotesError):
    """CUDA, cuBLAS или cuDNN недоступны."""


class OutputExistsError(MeetingNotesError):
    """Выходной файл уже существует, а --overwrite не указан."""


class CudaOutOfMemoryError(MeetingNotesError):
    """Не хватило видеопамяти GPU."""
