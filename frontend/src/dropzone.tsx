import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useUser } from '@clerk/clerk-react'

function DropZonePage() {
    const { isSignedIn, user } = useUser()

    const onDrop = useCallback( async (acceptedFiles: File[]) => {
        if (user) {
            for(let i = 0; i < acceptedFiles.length; i++) {
                const formData = new FormData()
                const payload = {video: acceptedFiles[i], id: user!.id}
                formData.append('payload', JSON.stringify(payload))

                const response = await fetch("/api/upload", {
                    method: "POST",
                    body: formData
                })

                console.log(response)
            }
        }
        else {
            console.log("User not logged in.")
        }
    }, [user])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
        onDrop, 
        accept: {
            'video/*': [],
        },
        maxSize: 3.5 * 100000000
    })

    return (
        <div className={"border-5"} {...getRootProps()}>
            <input {...getInputProps()} />
            {
                isDragActive ?
                    <p>Drag files here</p> :
                    <p>Drag and drop files, or select here to open file explorer</p>
            }
        </div>
    )
}


export default DropZonePage