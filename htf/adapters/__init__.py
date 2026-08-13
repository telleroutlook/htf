"""HTF adapter modules — optional bridges to third-party tensor backends.

Each adapter extracts a state vector from a backend-specific object and hands it
to :func:`htf.rayleigh_cert.rayleigh_certificate` to produce a
:class:`~htf.rayleigh_cert.RayleighCertificate`.  The adapters never run any
optimisation themselves — they only verify.

Available adapters
------------------
quimb_adapter    quimb MatrixProductState → RayleighCertificate
tenpy_adapter    TeNPy MatrixProductState → RayleighCertificate
"""
