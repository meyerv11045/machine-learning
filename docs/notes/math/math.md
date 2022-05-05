# Math for ML

## Inner Products

- aka dot products
- Given a euclidean plane with a line $L$ passing through the origin and a unit vector $w$ perpendicular to $L$ (the normal to the line)
- If you take any vector $x$, then the dot product $x \cdot w > 0$ if $x$ is on the same side of $L$ as $w$ and negative otherwise ($=0$ if its on the line)
- Inner product is a linear function (sum of scalar multiplication) so it can be optimized easily 
- The dot product as a decision rule in any $n$ dimensional space is useful for the Kernel Trick where nonlinearly separable data is trasnformed into a linearly separable space with no additional computational cost since all that is needed in the formulas to make a decision is the dot product 
- [Inner Product as Decision Rule](https://jeremykun.com/2017/05/22/the-inner-product-as-a-decision-rule/)

## Positive Definite

- Positive definition matrix $A$ has all positive eigenvalues
- a positive definite matrix means that for a vector $x$ in the first quadrant, applying the transformation to it $Ax$, would result in a vector still in the first quadrant
    - in other words the maximum angle between the two transformed vectors is < 90 degrees ($\pi /2$ rad)