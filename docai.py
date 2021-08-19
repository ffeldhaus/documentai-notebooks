from google.api_core.client_options import ClientOptions
from google.protobuf.json_format import MessageToJson
from google.cloud import documentai_v1beta3 as documentai
from google.cloud import storage

import ipywidgets as widgets

from PIL import Image, ImageDraw

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import resolve1

import humanize

from tqdm.notebook import tqdm

import mimetypes
import io
import os
import re
import json
from textwrap import indent
from pathlib import Path
import numpy as np
from enum import Enum
from types import SimpleNamespace

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    LINE = 4
    TOKEN = 5

class ProcessorLimits():
    def __init__(self, supported_file_types: list, max_pages_sync: int, max_pages_async: int, max_size_bytes_sync: int, max_size_bytes_async: int):
        self.supported_file_types = list(supported_file_type.casefold() for supported_file_type in supported_file_types)
        self.max_pages_sync = max_pages_sync
        self.max_pages_async = max_pages_async
        self.max_size_bytes_sync = max_size_bytes_sync
        self.max_size_bytes_async = max_size_bytes_async
    
    def __repr__(self):
        string = (
            f"supported_file_types: {self.supported_file_types}\n"
            f"max_pages_sync: {self.max_pages_sync}\n"
            f"max_pages_async: {self.max_pages_async}\n"
            f"max_size_bytes_sync: {humanize.naturalsize(self.max_size_bytes_sync, binary=True)}\n"
            f"max_size_bytes_async: {humanize.naturalsize(self.max_size_bytes_async, binary=True)}\n"
        )
        return string
    
class ProcessorDetails():
    def __init__(self, processor_id: str, processor_type: documentai.ProcessorType, processor_limits: ProcessorLimits):
        self.processor_id = processor_id
        self.processor_type = processor_type
        self.processor_limits = processor_limits
    
    def __repr__(self):
        string = (
            f"processor_id: {self.processor_id}\n"
            f"processor_type: {{\n{indent(str(self.processor_type),'  ',lambda line: True)}}}\n"
            f"processor_limits: {{\n{indent(str(self.processor_limits),'  ',lambda line: True)}}}"
        )
        return string
    
def get_client(location: str):
    client_options = ClientOptions(api_endpoint = f"{location}-documentai.googleapis.com")
    return documentai.DocumentProcessorServiceClient(client_options = client_options)

def list_processors(project_id: str, location: str):
    client = get_client(location = location)
    parent = f"projects/{project_id}/locations/{location}"
    return client.list_processors(parent = parent)

def get_processor(project_id: str, location: str, processor_id: str):
    processors = list_processors(project_id = project_id, location = location)
    return next((processor for processor in processors  if f"processors/{processor_id}" in processor.name), None)

def fetch_processor_types(project_id: str, location: str):
    client = get_client(location = location)
    parent = f"projects/{project_id}/locations/{location}"
    processor_types_response = client.fetch_processor_types(parent = parent)
    return processor_types_response.processor_types

def get_processor_details(project_id: str, location: str, processor_id: str):
    processor = get_processor(project_id = project_id, location = location, processor_id = processor_id)
    processor_types = fetch_processor_types(project_id = project_id, location = location)
    processor_type = next((processor_type for processor_type in processor_types if processor.type_ == processor_type.type_), None)
    with open('../processor_limits.json') as processor_limits_file:
         all_processor_limits = json.load(processor_limits_file)
    processor_limits = next((processor_limits for processor_limits in all_processor_limits if processor.type_ == processor_limits["type_"]), None)
    if processor_limits:
        processor_limits = ProcessorLimits(supported_file_types = processor_limits["supported_file_types"], max_pages_sync = processor_limits["max_pages_sync"], max_pages_async = processor_limits["max_pages_async"], max_size_bytes_sync = processor_limits["max_size_bytes_sync"], max_size_bytes_async = processor_limits["max_size_bytes_async"])
    processor_details = ProcessorDetails(
        processor_id = processor_id,
        processor_type = processor_type,
        processor_limits = processor_limits
    )
    return processor_details

def create_processor(project_id: str, location: str, type_: str, display_name: str, kms_key_name: str = ""):
    client = get_client(location = location)
    parent = f"projects/{project_id}/locations/{location}"
    processor = documentai.Processor(
        type_ = type_,
        display_name = display_name,
        kms_key_name = kms_key_name
    )
    result = client.create_processor(parent = parent, processor = processor)
    print(f"Processor {result.name} successfully created")
    return result

def delete_processor(project_id: str, location: str, processor_id: str):
    client = get_client(location = location)
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
    result = client.delete_processor(name = name)
    print(f"Processor {name} successfully deleted")

def disable_processor(project_id: str, location: str, processor_id: str):
    client = get_client(location = location)
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
    disable_processor_request = documentai.DisableProcessorRequest(name = name)
    return client.delete_processor(request = disable_processor_request)

def enable_processor(project_id: str, location: str, processor_id: str):
    client = get_client(location = location)
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
    enable_processor_request = documentai.EnableProcessorRequest(name = name)
    return client.delete_processor(request = enable_processor_request)

def process(project_id: str, location: str, processor_id: str, content: bytes, mime_type: str, skip_human_review: bool = False) -> dict:
    """Synchronous (online) process document using REST API.

    Processes document content with given mime type and blocks until result is returned.
    Optionally allows to skip human review if enabled for the processor.
    See details at
    https://cloud.google.com/document-ai/docs/reference/rest/v1/projects.locations.processors/process

    Args:
        project_id: GCP Project ID (e.g. 940142200552).
        location: Location to be used for the request (e.g. eu or us)
        processor_id: Processor ID (e.g. 7f9cd174a388594a).
        content: Document content as byte string.
        mime_type: An IANA MIME type (RFC6838).
        skip_human_review: Optional; Whether Human Review feature should be skipped for this request. Default to false.

    Returns:
        A dict containing processed document and human_review_status.
        See details at
        https://cloud.google.com/document-ai/docs/reference/rest/v1/ProcessResponse
    """
    
    client = get_client(location = location)

    processor = get_processor(project_id = project_id, location = location, processor_id = processor_id)
    if not processor:
        raise Exception(f"Processor with ID {processor_id} not found")

    processor_details = get_processor_details(project_id = project_id, location = location, processor_id = processor_id)

    # check if request is within supported limits
    if processor_details.processor_limits:
        content_size = len(content)
        if content_size > processor_details.processor_limits.max_size_bytes_sync:
            raise Exception(f"Content of size {humanize.naturalsize(content_size, binary=True)} is larger than the {humanize.naturalsize(processor_details.processor_limits.max_size_bytes_sync, binary=True)} limit of synchronous processing, please use batch processing.")

        page_count = 1
        
        if mimetypes.guess_extension(mime_type)[1:] not in processor_details.processor_limits.supported_file_types:
            raise Exception(f"MIME type {mime_type} not supported by processor")

        if mime_type == "image/tiff":
            page_count = Image.open(io.BytesIO(content)).n_frames

        if mime_type == "application/pdf":
            parser = PDFParser(io.BytesIO(content))
            document = PDFDocument(parser)

            # This will give you the count of pages
            page_count = resolve1(document.catalog['Pages'])['Count']

        if page_count > processor_details.processor_limits.max_pages_sync:
            raise Exception(f"Page count of {page_count} is larger than {processor_details.processor_limits.max_pages_sync} page limit of synchronous processing, please use batch processing.")
    else:
        raise Warining(f"Processor details for processor with ID {processor_id} not found")

    # Create raw document from image content
    raw_document = documentai.RawDocument(
        content = content,
        mime_type = mime_type
    )

    # Process document
    process_request = documentai.ProcessRequest(
        name = processor.name,
        raw_document = raw_document,
        skip_human_review = skip_human_review
    )

    return client.process_document(request=process_request)

def process_gcs_uri(project_id: str, location: str, processor_id: str, gcs_uri: str, skip_human_review: bool = False):
    # Instantiate a Google Cloud Storage Client
    storage_client = storage.Client()
    if not gcs_uri.startswith("gs://"):
        raise Exception(f"gcs_uri {gcs_uri} missing gs:// prefix.")
    
    mime_type = mimetypes.guess_type(gcs_uri)[0]
    if not mime_type:
        raise Exception(f"MIME type of gcs_uri {gcs_uri} could not be guessed from file extension.")
    processor_details = get_processor_details(project_id = project_id, location = location, processor_id = processor_id)
    if processor_details.processor_limits:
        if mimetypes.guess_extension(mime_type)[1:].casefold() not in processor_details.processor_limits.supported_file_types:
            raise Exception(f"MIME type {mime_type} of {gcs_uri} not supported by processor")
    
    blob = storage.Blob.from_string(uri = gcs_uri, client = storage_client)
    image_content = blob.download_as_string()    
    
    result = process(project_id = project_id, location = location, processor_id = processor_id, content = image_content, mime_type = mime_type)
    result.document.uri = gcs_uri
    return result

def process_gcs_bucket(project_id: str, location: str, processor_id: str, bucket: str, prefix: str = "", skip_human_review: bool = False) -> list:
    # Instantiate a Google Cloud Storage Client
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket, prefix = prefix)
    results = []
    pbar = tqdm(list(blobs),unit = "document")
    for blob in pbar:
        if not blob.name.endswith('/'):
            gcs_uri = f"gs://{bucket}/{blob.name}"            
            pbar.set_postfix({"document": gcs_uri})
            pbar.refresh()
            try:
                results.append(process_gcs_uri(project_id = project_id, location = location, processor_id = processor_id, gcs_uri = gcs_uri, skip_human_review = skip_human_review))
            except Exception as e: 
                tqdm.write("\x1b[31m" + str(e) + "\x1b[0m")
    return results

def process_file(project_id: str, location: str, processor_id: str, path: str, skip_human_review: bool = False):
    path = Path(path)
    if not path.is_file():
        raise Exception(f"Path {path} is not a file")
    
    mime_type = mimetypes.guess_type(path.name)[0]
    processor_details = get_processor_details(project_id = project_id, location = location, processor_id = processor_id)
    if path.suffix[1:].casefold() not in (supported_file_type.casefold() for supported_file_type in processor_details.processor_limits.supported_file_types):
        raise Exception(f"MIME type {mime_type} of {gcs_uri} not supported by processor")
    image_content = open(path, "rb").read()
    result = process(project_id = project_id, location = location, processor_id = processor_id, content = image_content, mime_type = mime_type, skip_human_review = skip_human_review)
    result.document.uri = path.resolve().as_uri()
    return result
    
def process_dir(project_id: str, location: str, processor_id: str, path: str, skip_human_review: bool = False):
    path = Path(path)
    results = []
    if not path.exists():
        raise(f"Directory {path} does not exist")
    if not path.is_dir():
        raise Exception(f"{path} is not a directory")
        
    files = path.rglob('*')
    pbar = tqdm(list(files), unit = "document")
    for path in pbar:
        pbar.set_postfix({"document": path.resolve().as_uri()})
        pbar.refresh()
        try:
            results.append(process_file(project_id = project_id, location = location, processor_id = processor_id, path = path, skip_human_review = skip_human_review))
        except Exception as e: 
            pbar.write("\x1b[31m" + str(e) + "\x1b[0m")  
    
    return results
        
def get_page_bounds(page, feature):
    """Returns document bounds given the OCR output page."""

    bounds = []

    # Collect specified feature bounds by enumerating all document features
    if (feature == FeatureType.BLOCK):
        for block in page.blocks:
            if not block.layout.bounding_poly.vertices:
                block.layout.bounding_poly.vertices = []
                for normalized_vertice in block.layout.bounding_poly.normalized_vertices:
                    block.layout.bounding_poly.vertices.append(documentai.Vertex(x=int(normalized_vertice.x * page.image.width),y=int(normalized_vertice.y * page.image.height)))
            bounds.append(block.layout.bounding_poly)
    if (feature == FeatureType.PARA):
        for paragraph in page.paragraphs:
            if not paragraph.layout.bounding_poly.vertices:
                paragraph.layout.bounding_poly.vertices = []
                for normalized_vertice in paragraph.layout.bounding_poly.normalized_vertices:
                    paragraph.layout.bounding_poly.vertices.append(documentai.Vertex(x=int(normalized_vertice.x * page.image.width),y=int(normalized_vertice.y * page.image.height)))
            bounds.append(paragraph.layout.bounding_poly)
    if (feature == FeatureType.LINE):        
        for line in page.lines:
            if not line.layout.bounding_poly.vertices:
                line.layout.bounding_poly.vertices = []
                for normalized_vertice in line.layout.bounding_poly.normalized_vertices:
                    line.layout.bounding_poly.vertices.append(documentai.Vertex(x=int(normalized_vertice.x * page.image.width),y=int(normalized_vertice.y * page.image.height)))
            bounds.append(line.layout.bounding_poly)
    if (feature == FeatureType.TOKEN):        
        for token in page.tokens:
            if not token.layout.bounding_poly.vertices:
                token.layout.bounding_poly.vertices = []
                for normalized_vertice in token.layout.bounding_poly.normalized_vertices:
                    token.layout.bounding_poly.vertices.append(documentai.Vertex(x=int(normalized_vertice.x * page.image.width),y=int(normalized_vertice.y * page.image.height)))
            bounds.append(token.layout.bounding_poly)


    # The list `bounds` contains the coordinates of the bounding boxes.
    return bounds

def draw_boxes(image, bounds, color, width):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        points = (
            (bound.vertices[0].x, bound.vertices[0].y),
            (bound.vertices[1].x, bound.vertices[1].y),
            (bound.vertices[2].x, bound.vertices[2].y),
            (bound.vertices[3].x, bound.vertices[3].y),
            (bound.vertices[0].x, bound.vertices[0].y)
        )
        draw.line(points,fill=color,width=width,joint='curve')
    return image

def render_ocr_page(page, block = True, para = True, line = True, token = True):  
    image = Image.open(io.BytesIO(page.image.content))

    # this will draw the bounding boxes for block, paragraph, line and token
    if block:
        bounds = get_page_bounds(page, FeatureType.BLOCK)
        draw_boxes(image, bounds, color='blue', width=8)
    if para:
        bounds = get_page_bounds(page, FeatureType.PARA)
        draw_boxes(image, bounds, color='red',width=6)
    if line:
        bounds = get_page_bounds(page, FeatureType.LINE)
        draw_boxes(image, bounds, color='yellow',width=4)
    if token:
        bounds = get_page_bounds(page, FeatureType.TOKEN)
        draw_boxes(image, bounds, color='green',width=2)

    image.show()

    # uncomment if you want to save the image with bounding boxes locally
    #image.save(document.name)
    
def display_ocr_output(output: list):
    dropdown = widgets.Dropdown(options=output)
    block = widgets.ToggleButton(description='BLOCK', value=True)
    para = widgets.ToggleButton(description='PARA', value=False)
    line = widgets.ToggleButton(description='LINE', value=False)
    token = widgets.ToggleButton(description='TOKEN', value=True)

    ui = widgets.HBox([dropdown, block, para, line, token])

    def show_page(page, block,para,line,token):
        render_ocr_page(page,block,para,line,token)

    out = widgets.interactive_output(show_page, {'page':dropdown, 'block':block,'para':para,'line':line,'token':token})

    display(ui, out)