import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

function DropZonePage() {
    const onDrop = useCallback( async (acceptedFiles: File[]) => {
        for(let i = 0; i < acceptedFiles.length; i++) {
            const response = await fetch("/api/videos", {
                method: "POST",
                body: acceptedFiles[i],
            })

            console.log(response)
        }
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
        onDrop, 
        accept: {
            'video/*': [],
        },
        maxSize: 3.5 * 100000000
    })

    return (
        <form className="border">
            <div {...getRootProps()}>
                <input {...getInputProps()} />
                {
                    isDragActive ?
                        <p>Drag files here</p> :
                        <p>Drag and drop files, or select here to open file explorer</p>
                }
            </div>
        </form>
    )
}


export default DropZonePage