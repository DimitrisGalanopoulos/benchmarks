#!/usr/bin/awk -f

BEGIN {
    LINT = "fatal"
}

BEGINFILE {
    printf("matrix_name,num_iterations,error\n")
}


function dirname(path)
{
    path = gensub(/\/+/, "/", "g", path)
    sub(/[^/]+\/?$/, "", path)
    if (path == "/") return "/"
    sub(/\/$/, "", path)
    return (path == "") ? "." : path
}

function basename(path)
{
    path = gensub(/\/+/, "/", "g", path)
    sub(/\/$/, "", path)
    sub(/^.*\//, "", path)
    return path
}


/File: / {
    path = $2
    filename = basename(path)
    error_best_current = 1e20
}

/k = / {
    # k = 0          error = 975.8        error_explicit = 975.8        error_best = 975.8
    k = $3
    error = $6
    error_explicit = $9
    error_best = $12
    if (error_best < error_best_current)
    {
        printf("%s,%d,%s\n", filename, k, error_best)
        error_best_current = error_best
    }
}

ENDFILE {
}

END {
}



