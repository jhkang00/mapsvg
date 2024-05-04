function shifthemi(origS, pixdis)
    newS = replace(origS, r"(([.]|\d)+)(\s)(([.]|\d)+)" => x -> "$(parse(Float64,split(x)[1])+pixdis) $(split(x)[2])");
    return newS
end