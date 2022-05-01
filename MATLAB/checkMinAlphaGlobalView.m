function minimum = checkMinAlphaGlobalView(group, aol)
    minimum = -1;
    for alpha = 0: 3: 90
        if checkGlobalView(group, aol, alpha)
            minimum = alpha;
            return;
        end
    end
end
